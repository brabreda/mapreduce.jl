# module mapreduce
using KernelAbstractions
using GPUArrays

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()


@kernel function partial_mapreduce_grid(f, op, neutral, strideSize, groupsize, localReduceIndices, R, As...)
  global_Idx = @index(Local, Linear) + (@index(Group, Linear) - 1) * prod(@groupsize)
  Iout = @index(Group, Cartesian)
  Iother = CartesianIndex(Tuple(Iout)[1:(length(Iout)-1)]..., 1)

  # load the neutral value
  @inbounds neutral = if neutral === nothing
      R[Iout]
  else
      neutral
  end

  val = op(neutral, neutral)

  # reduce serially across chunks of input vector that don't fit in a block
  ireduce = mod1( global_Idx, strideSize)
  @inbounds while ireduce <= length(localReduceIndices)
      Ireduce = localReduceIndices[ireduce]
      J = max(Iother, Ireduce)
      val = op(val, f(_map_getindex(As,J)...))
      ireduce += strideSize
  end

  val = @groupreduce(op, val, neutral, groupsize)

  # write back to memory
  if @index(Local, Linear) == 1
    R[Iout] = val
  end 
end

function mapreducedim(f::F, op::OP, R::AnyGPUArray,
                               A::Union{AbstractArray,Broadcast.Broadcasted};
                               init=nothing) where {F, OP}
  Base.check_reducedims(R, A)
  length(A) == 0 && return R # isempty(::Broadcasted) iterates
  KABackend = get_backend(A) 

  # add singleton dimensions to the output container, if needed
  if ndims(R) < ndims(A)
      dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
      R = reshape(R, dims)
  end

  # Interation domain, the indices of the iteration space are split into two parts. localReduceIndices
  # covers the part of the indices that is identical for every group, the other part deduced form KA.
  # @index(Group, Cartesian) covers the part of the indices that is different for every group.
  localReduceIndices = CartesianIndices((ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A))..., Base.OneTo(1), Base.OneTo(1)))

  ndrange = (ifelse.(axes(A) .== axes(R), size(A), 1)..., length(localReduceIndices), 1)
  groupsize = (ones(Int, ndims(A))..., length(localReduceIndices), 1)

  # allocate an additional, empty dimension to write the reduced value to.
  # this does not affect the actual location in memory of the final values,
  # but allows us to write a generalized kernel supporting partial reductions.
  R′ = reshape(R, (size(R)..., 1, 1))

  # we use val() to make the groupsize a compile-time constant.
  args = (f, op, init, 1, Val(1), localReduceIndices, R′, A)
  kernelObj = partial_mapreduce_grid(KABackend)
  max_groupsize, max_ndrange = launch_config(kernelObj, args...; workgroupsize=groupsize, ndrange=ndrange)
  max_groupsize = min(max_groupsize, length(localReduceIndices))

  ndrange = (ifelse.(axes(A) .== axes(R), size(A), 1)..., max_groupsize, 1)
  groupsize = (ones(Int, ndims(A))...,max_groupsize, 1)

  groups = if prod(ndrange) <=  max_ndrange 
    min(fld((max_ndrange ÷ max_groupsize), prod(ndrange) ÷ prod(groupsize)),  # are there groups left?
        cld(length(localReduceIndices), max_groupsize))                  # how many groups do we want?
  else 
    1
  end

  ndrange = (ifelse.(axes(A) .== axes(R), size(A), 1)..., max_groupsize, groups)
  stridesize = max_groupsize * groups

  # If we have only one group per slice, every slice can be reduced in one go, no second kernel is needed.
  if groups == 1
    partial_mapreduce_grid(KABackend)( f, op, init, stridesize, Val(prod(groupsize)), localReduceIndices, R′, A, workgroupsize=groupsize, ndrange=ndrange)
  else
      # we need temporary storage to hold partial reductions for every slice the endresult can be calcultated
      # by reducing the temporary storage in the direction of the every slice.
      partial = similar(R, (size(R)..., 1, groups))
      if init === nothing
          # without an explicit initializer we need to copy from the output container
          partial .= R
      end
  
      # NOTE: we can't use the previously-compiled kernel, since the type of `partial`
      #       might not match the original output container (e.g. if that was a view).
      partial_mapreduce_grid(KABackend)(f, op, init, stridesize, Val(prod(groupsize)), localReduceIndices, partial, A, workgroupsize=groupsize, ndrange=ndrange)
      mapreducedim(identity, op, R′, partial; init=init)
  end

  return R
end