# module mapreduce
using CUDA
using KernelAbstractions
#using GPUArrays

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()

# Reduce an array across the grid. All elements to be processed can be addressed by the
# product of the two iterators `Rreduce` and `Rother`, where the latter iterator will have
# singleton entries for the dimensions that should be reduced (and vice versa).
@kernel function partial_mapreduce_grid(f, op, neutral, strideSize, localReduceIndices, R, As...)
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

  val = @groupreduce(op, val, neutral)

  # write back to memory
  if @index(Local, Linear) == 1
    R[Iout] = val
  end 
end

function mapreducedim(f::F, op::OP, R,
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

  ndrange = (size(A)..., 1, 1)
  groupsize = (ones(Int, ndims(A))..., 1, 1)

  # Interation domain, the indices of the iteration space are split into two parts. localReduceIndices
  # covers the part of the indices that is identical for every group, the other part deduced form KA.
  # @index(Group, Cartesian) covers the part of the indices that is different for every group.
  localReduceIndices = CartesianIndices((ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A))..., Base.OneTo(1), Base.OneTo(1)))

  # allocate an additional, empty dimension to write the reduced value to.
  # this does not affect the actual location in memory of the final values,
  # but allows us to write a generalized kernel supporting partial reductions.
  R′ = reshape(R, (size(R)..., 1, 1))

  # we create dummy dimensions for the group and ndrange that allows us the determine a launch 
  # configuration. Making the dimensions one size bigger allows us to create the ideal groupsize
  # in this dimension.

  args = (f, op, init, 1, localReduceIndices, R′, A)
  kernelObj = partial_mapreduce_grid(KABackend)
  max_groupsize, max_ndrange = launch_config(kernelObj, args...; workgroupsize=groupsize, ndrange=ndrange)

  # Instead of using KA's indices, we use extern CartesianIndices. This allows use more indices per 
  # group than allowed by hardware + we can add the dimensions of the group to the end and use Linear
  # indexing.

  # ndrange = (ifelse.(groupsize .== ndrange, ndrange, 1)..., launch_groupsize)
  # groupsize = (ones(Int, length(groupsize))..., launch_groupsize)

  strideSize = max_groupsize
  oneStepReduce = true
  RotherSize = prod(size(R))

  
  if prod(size(R)) == 1
    launch_ndrange = ifelse(prod(size(A)) > max_ndrange, max_ndrange, prod(size(A)))
    
    ndrange = (ones(Int, length(ndrange)-2)..., max_groupsize, cld(launch_ndrange, max_groupsize))  
    groupsize = (ones(Int, length(groupsize)-2)...,  max_groupsize, 1) 
    oneStepReduce = max_groupsize >= prod(ndrange)
    strideSize = prod(ndrange)
  else
    ndrange = (ifelse.(size(A) .== size(R), size(A), 1)..., max_groupsize, 1)
    groupsize = (ones(Int, length(groupsize)-2)..., max_groupsize, 1)
    RotherSize = 1
  end

  # If we have only one group per slice, every slice can be reduced in one go, no second kernel is needed.
  if oneStepReduce
    partial_mapreduce_grid(KABackend, groupsize)( f, op, init, strideSize, localReduceIndices, R, A, ndrange=ndrange)
  else
      # we need temporary storage to hold partial reductions for every slice the endresult can be calcultated
      # by reducing the temporary storage in the direction of the every slice.
      partial = similar(R, (size(R)..., 1, cld(prod(ndrange), prod(groupsize))))
      if init === nothing
          # without an explicit initializer we need to copy from the output container
          partial .= R
      end

      # NOTE: we can't use the previously-compiled kernel, since the type of `partial`
      #       might not match the original output container (e.g. if that was a view).
      partial_mapreduce_grid(KABackend, groupsize)(f, op, init, strideSize, localReduceIndices, partial, A, ndrange=ndrange)
      display(partial)
      mapreducedim(identity, op, R′, partial; init=init)
  end

  return R
end

  # # Every slices is reduced by one group, if there are groups leftover a slice can be reduced by multiple groups. 
  # # The result can then be combined with in a second kernellaunch 
  # slice_groups = prod(size(R))
  # groups_per_slice = if launch_groupsize <= prod(groupsize)
  #     1                                        # We don't need multiple groups to cover 1 slice.
  # else
  #     min(cld(length(localReduceIndices), launch_groupsize), # it can be optimal to use multiple groups for one slice
  #         cld(launch_ndrange, prod(groupsize)))                # but we should not use more groups than we have.
  # end  

  # total_groups = slice_groups * slice_groups