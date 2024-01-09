# module mapreduce
using KernelAbstractions
using NVTX

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()

# Reduce an array across the grid. All elements to be processed can be addressed by the
# product of the two iterators `Rreduce` and `Rother`, where the latter iterator will have
# singleton entries for the dimensions that should be reduced (and vice versa).
@kernel function nd_mapreduce_grid(f, op, neutral, Rreduce, Rother, R, As...)
  threadIdx_reduce = @index(Local, Linear)
  blockDim_reduce = prod(@groupsize())
  blockIdx_reduce, blockIdx_other = fldmod1(@index(Group,Linear), length(Rother))
  gridDim_reduce = prod(@ndrange) ÷ length(Rother)

  # block-based indexing into the values outside of the reduction dimension
  # (that means we can safely synchronize threads within this block)
  iother = blockIdx_other
  @inbounds if iother <= length(Rother)
      Iother = Rother[iother]

      # load the neutral value
      Iout = CartesianIndex(Tuple(Iother)..., blockIdx_reduce)
      neutral = if neutral === nothing
          R[Iout]
      else
          neutral
      end

      val = op(neutral, neutral)

      # reduce serially across chunks of input vector that don't fit in a block
      ireduce = threadIdx_reduce + (blockIdx_reduce - 1) * blockDim_reduce
      while ireduce <= length(Rreduce)
          Ireduce = Rreduce[ireduce]
          J = max(Iother, Ireduce)
          val = op(val, f(_map_getindex(As, J)...))
          ireduce += blockDim_reduce * gridDim_reduce
      end

      val =  @groupreduce(op, val, neutral)

      # write back to memory
      if threadIdx_reduce == 1
          R[Iout] = val
      end
  end
end

@kernel function scalar_mapreduce_grid(f, op, neutral, R, A)
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

      # every thread reduces a few values parrallel
  index = threadIdx_global 
  while index <= length(A)
      val = op(val,f(A[index]))
      index += gridsize
  end

      # reduce every block to a single value
  val = @groupreduce(op, val, neutral)

      # use helper function to deal with atomic/non atomic reductions
  if threadIdx_local == 1
    @inbounds R[groupIdx] = val
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

  # allocate an additional, empty dimension to write the reduced value to.
  # this does not affect the actual location in memory of the final values,
  # but allows us to write a generalized kernel supporting partial reductions.
  R′ = reshape(R, (size(R)..., 1))

  if length(R) == 1
    groupsize = 1 
    ndrange = 1

    args = (f, op, init, R′, A)
    kernelObj = scalar_mapreduce_grid(KABackend)
    max_groupsize, max_ndrange = launch_config(kernelObj, args...; workgroupsize=groupsize, ndrange=ndrange)

    groupsize = max_groupsize
    ndrange = min(length(A), max_ndrange)

    if length(A) <= max_groupsize
      scalar_mapreduce_grid(KABackend)( f, op, init, R′, A, ndrange=groupsize, workgroupsize=groupsize)
    else
      partial = similar(R, (size(R)..., cld(ndrange, groupsize)))
      if init === nothing
          # without an explicit initializer we need to copy from the output container
          partial .= R
      end
      
      scalar_mapreduce_grid(KABackend)( f, op, init, partial, A, ndrange=ndrange, workgroupsize=groupsize)
  
      mapreducedim(identity, op, R', partial; init=init)
    end
  else 

    # Interation domain, the indices of the iteration space are split into two parts. localReduceIndices
    # covers the part of the indices that is identical for every group, the other part deduced form KA.
    # @index(Group, Cartesian) covers the part of the indices that is different for every group.
    localReduceIndices = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))
    sliceIndices = CartesianIndices(axes(R))

    ndrange = (ifelse.(size(A) .== size(R), size(A), 1)..., length(localReduceIndices))
    groupsize = (ones(Int, ndims(A))..., length(localReduceIndices))

    args = (f, op, init, localReduceIndices, sliceIndices, R′, A)
    kernelObj = nd_mapreduce_grid(KABackend)
    max_groupsize, max_ndrange = launch_config(kernelObj, args...; workgroupsize=(prod(groupsize)), ndrange=(prod(ndrange)))

    # Instead of using KA's indices, we use extern CartesianIndices. This allows use more indices per 
    # group than allowed by hardware + we can add the dimensions of the group to the end and use Linear
    # indexing.

    ndrange = (ifelse.(size(A) .== size(R), size(A), 1)..., max_groupsize)
    groupsize = (ones(Int, length(groupsize)-1)..., max_groupsize)

    groups = if prod(ndrange) <= max_ndrange 
    max_groupsize, max_ndrange = launch_config(kernelObj, args...; workgroupsize=groupsize, ndrange=ndrange)
      min(fld((max_ndrange ÷ max_groupsize), (prod(ndrange) ÷ max_groupsize)),  # are there groups left?
          cld(length(localReduceIndices), max_groupsize))                  # how many groups do we want?
    else 
      1
    end

    ndrange = (ifelse.(axes(A) .== axes(R), size(A), 1)..., max_groupsize*groups)


    if groups == 1
      nd_mapreduce_grid(KABackend)(f, op, init, localReduceIndices,sliceIndices, R′, A, ndrange=(prod(ndrange)), workgroupsize=(prod(groupsize)))
    else
      partial = similar(R, (size(R)..., groups))
      if init === nothing
          # without an explicit initializer we need to copy from the output container
          partial .= R
      end

      nd_mapreduce_grid(KABackend)(f, op, init, localReduceIndices, sliceIndices, partial, A, ndrange=(prod(ndrange)), workgroupsize=(prod(groupsize)))

      mapreducedim(identity, op, R′, partial; init=init)
    end
  end
  return R
end