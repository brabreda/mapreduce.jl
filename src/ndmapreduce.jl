# module mapreduce
using KernelAbstractions
using GPUArrays

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()

@kernel function nd_mapreduce_grid(f, op, neutral, groupsize::Val{GROUPSIZE}, localReduceIndices, sliceIndices, R, As...) where {GROUPSIZE}
  threadIdx_local = @index(Local, Linear)
  sliceIdx, Idx_in_slice = fldmod1(@index(Group,Linear), length(sliceIndices))
  elements_per_group = prod(@ndrange) ÷ length(sliceIndices)

  iother = Idx_in_slice
  @inbounds if iother <= length(sliceIndices)
      Iother = sliceIndices[iother]

      Iout = CartesianIndex(Tuple(Iother)..., sliceIdx)
      neutral = if neutral === nothing
          R[Iout]
      else
          neutral
      end

      val = op(neutral, neutral)

      # reduce serially across chunks of input vector that don't fit in a block
      ireduce = threadIdx_local + (sliceIdx - 1) * GROUPSIZE
      while ireduce <= length(localReduceIndices)
          Ireduce = localReduceIndices[ireduce]
          J = max(Iother, Ireduce)
          val = op(val, f(_map_getindex(As, J)...))
          ireduce += elements_per_group * GROUPSIZE
      end

      val =  @groupreduce(op, val, neutral, groupsize)

      # write back to memory
      if threadIdx_local == 1
          R[Iout] = val
      end
  end
end

@kernel function scalar_mapreduce_grid(f, op, neutral, groupsize, R, A)
  threadIdx_local = @index(Local)
  threadIdx_global = @index(Global)
  groupIdx = @index(Group)
  gridsize = prod(@ndrange())

  # load neutral value
  neutral = if neutral === nothing
      R[1]
  else
      neutral
  end
  
  val = op(neutral, neutral)

  # every thread sequentially reduces a chunk if possible
  index = threadIdx_global 
  while index <= length(A)
      val = op(val,f(A[index]))
      index += gridsize
  end

  # reduce the values of the group to a single value
  val = @groupreduce(op, val, neutral, groupsize)

  if threadIdx_local == 1
    @inbounds R[groupIdx] = val
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

  R′ = reshape(R, (size(R)..., 1))

  if length(R) == 1
    # allocate an additional, empty dimension to write the reduced value to.
    # this does not affect the actual location in memory of the final values,
    # but allows us to write a generalized kernel supporting partial reductions.
    ndrange = length(A)
    groupsize = 1 # we use one so we are sure to not use to much local memory

    args = (f, op, init, Val(groupsize), R′, A)
    kernelObj = scalar_mapreduce_grid(KABackend)
    max_groupsize, max_ndrange = launch_config(kernelObj, args...; workgroupsize=groupsize, ndrange=ndrange)

    ndrange = min(length(A), max_ndrange)
    groupsize = max_groupsize

    if length(A) <= max_groupsize
      scalar_mapreduce_grid(KABackend)( f, op, init, Val(groupsize), R′, A, ndrange=groupsize, workgroupsize=groupsize)
    else
      partial = similar(R, (size(R)..., cld(ndrange, groupsize)))
      if init === nothing
          # without an explicit initializer we need to copy from the output container
          partial .= R
      end
      
      scalar_mapreduce_grid(KABackend)( f, op, init, Val(groupsize), partial, A, ndrange=ndrange, workgroupsize=groupsize)
  
      mapreducedim(identity, op, R', partial; init=init)
    end
  else 
    # Interation domain, the indices of the iteration space are split into two parts. localReduceIndices
    # covers the part of the indices that is identical for every group, the other part deduced form KA.
    # @index(Group, Cartesian) covers the part of the indices that is different for every group.
    localReduceIndices = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))
    sliceIndices = CartesianIndices(axes(R))

    ndrange = length(localReduceIndices) * length(sliceIndices)
    groupsize = length(localReduceIndices)

    args = (f, op, init, Val(1), localReduceIndices, sliceIndices, R′, A)
    kernelObj = nd_mapreduce_grid(KABackend)
    max_groupsize, max_ndrange = launch_config(kernelObj, args...; workgroupsize=groupsize, ndrange=ndrange)

    # Instead of using KA's indices, we use extern CartesianIndices. This allows use more indices per 
    # group than allowed by hardware + we can add the dimensions of the group to the end and use Linear
    # indexing
    groupsize = min(length(localReduceIndices), max_groupsize)
    ndrange = groupsize * length(sliceIndices)

    groups = if ndrange <= max_ndrange 
      min(fld((max_ndrange ÷ groupsize), (ndrange ÷ groupsize)),  # are there groups left?
          cld(length(localReduceIndices), groupsize))                  # how many groups do we want?
    else 
      1
    end

    ndrange = ndrange * groups
    
    if groups == 1
      nd_mapreduce_grid(KABackend)(f, op, init, Val(groupsize), localReduceIndices,sliceIndices, R′, A, ndrange=ndrange, workgroupsize=groupsize)
    else
      partial = similar(R, (size(R)..., groups))
      if init === nothing
          # without an explicit initializer we need to copy from the output container
          partial .= R
      end

      nd_mapreduce_grid(KABackend)(f, op, init, Val(groupsize), localReduceIndices, sliceIndices, partial, A, ndrange=ndrange, workgroupsize=groupsize)

      mapreducedim(identity, op, R′, partial; init=init)
    end
  end
  return R
end