
# module mapreduce
using CUDA
using KernelAbstractions
using GPUArrays

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()

# Reduce an array across the grid. All elements to be processed can be addressed by the
# product of the two iterators `Rreduce` and `Rother`, where the latter iterator will have
# singleton entries for the dimensions that should be reduced (and vice versa).
@kernel function partial_mapreduce_grid(f, op, neutral, R, As...)
  # decompose the 1D hardware indices into separate ones for reduction (across threads
  # and possibly blocks if it doesn't fit) and other elements (remaining blocks)
  threadIdx_reduce = @index(Local)
  blockDim_reduce = prod(@groupsize())
  groups = cld(prod(@ndrange()),prod(@groupsize()))
  blockIdx_reduce, blockIdx_other = fldmod1(@index(Group), groups)
  gridDim_reduce = prod(@ndrange()) ÷ blockDim_reduce ÷ groups

  # block-based indexing into the values outside of the reduction dimension
  # (that means we can safely synchronize threads within this block)
  Iother = @index(Group, Cartesian)

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
  while ireduce <= prod(@groupsize())
      J = @index(Global,Cartesian)
      val = op(val, f(_map_getindex(As, J)...))
      ireduce += blockDim_reduce * gridDim_reduce
  end

  val = @groupreduce(op, val, neutral)

  # write back to memory
  if threadIdx_reduce == 1
    R[Iout] = val
  end 
end

@kernel function big_mapreduce_kernel(f, op, neutral, Rreduce, Rother, R, As)
  grid_idx = @index(Global)
  @inbounds if grid_idx <= length(Rother)
      Iother = Rother[grid_idx]

      # load the neutral value
      neutral = if neutral === nothing
          R[Iother]
      else
          neutral
      end

      val = op(neutral, neutral)

      Ibegin = Rreduce[1]
      for Ireduce in Rreduce
          val = op(val, f(As[Iother + Ireduce - Ibegin]))
      end
      R[Iother] = val
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

  # iteration domain, one part covers the reductions that every group needs to reduce,
  # another part covers in what direction the groups should reduce
  workgroupDim = ifelse.(size(A) .== size(R), 1, size(A))
  ndrangeDim = size(A)

  # allocate an additional, empty dimension to write the reduced value to.
  # this does not affect the actual location in memory of the final values,
  # but allows us to write a generalized kernel supporting partial reductions.
  R′ = reshape(R, (size(R)..., 1))

  # how many threads do we want?
  #
  # threads in a block work together to reduce values across the reduction dimensions;
  # we want as many as possible to improve algorithm efficiency and execution occupancy.
  wanted_threads = prod(workgroupDim)

  # how many threads can we launch?
  #
  # we might not be able to launch all those threads to reduce each slice in one go.
  # that's why each threads also loops across their inputs, processing multiple values
  # so that we can span the entire reduction dimension using a single thread block.
  partial = similar(R, (size(R)..., 1))

  kernelObj = partial_mapreduce_grid(KABackend)
  kernel, ctx, ngroups, workgroupsize = launch_config(kernelObj, (f, op, init, R′, A)...; workgroupsize=workgroupDim, ndrange=ndrangeDim)

  other_blocks = prod(size(R))
  reduce_blocks = if other_blocks >= ngroups
      1
  else
      min(cld(prod(workgroupDim), workgroupsize), cld(ngroups, other_blocks)) # maximize occupancy
  end
  
  wanted_groups = reduce_blocks*other_blocks
  # @show wanted_groups
  # @show prod(workgroupDim)
  # if ngroups * workgroupsize < wanted_groups * prod(workgroupDim)
  #   println("big_mapreduce_kernel")
  #   Rother = CartesianIndices(axes(R))
  #   big_mapreduce_kernel(KABackend, 1024)(f, op, init, Rreduce, Rother, R′, A; workgroupsize=1024, ndrange=wanted_groups)
  #   return R
  # end

  # perform the actual reduction
  if reduce_blocks == 1
      # we can do the entire reduction in one go
      kernel(ctx, f, op, init, R′, A; threads=workgroupsize, blocks=wanted_groups)
  else
      # we need multiple steps to cover all values to reduce
      partial = similar(R, (size(R)..., ngroups))
      if init === nothing
          # without an explicit initializer we need to copy from the output container
          partial .= R
      end
      # NOTE: we can't use the previously-compiled kernel, since the type of `partial`
      #       might not match the original output container (e.g. if that was a view).
      println("partial_mapreduce_grid")
      kernel(ctx, f, op, init, partial, A; threads=workgroupsize, blocks=ngroups)

      mapreducedim(identity, op, R′, partial; init=init)
  end

  return R
end

# launch configuration
function launch_config(kernelObj, args...; workgroupsize, ndrange)
  backend = KernelAbstractions.backend(kernelObj)
  ndrange, workgroupsize, iterspace, dynamic = KernelAbstractions.launch_config(kernelObj, ndrange,workgroupsize)
  ctx = KernelAbstractions.mkcontext(kernelObj, ndrange, iterspace)
  @show ndrange

  kernel = CUDA.@cuda launch=false always_inline=backend.always_inline kernelObj.f(ctx, args...)
  return kernel, ctx, CUDA.launch_configuration(kernel.fun; max_threads=1024)...
end

using BenchmarkTools
# end

a = CUDA.randn(10,10,10,10)
b = similar(a,(10,10))
mapreduce(x->x, +, a, dims=(3,4)) == mapreducedim(x->x, +, b,a; init=Float32(0.0))



  # if CUDA.threadIdx().x == 25 && CUDA.blockIdx().x == 3
  #   Ireduce = Rreduce[ireduce]
  #   Iother = Rother[iother]
  #   #iii = @index(Global, Cartesian)
  #   #i = Iother
  #   #J = max(i, ii)

  #   #groupindex = @index(Group, Cartesian)
  #   index = @index(Group, Cartesian)

  #   CUDA.@cuprintln("index: \t\t" , index[1], "\t", index[2], "\t", index[3])

    
  #   CUDA.@cuprintln("Iother: \t" , Iother[1], "\t", Iother[2], "\t", Iother[3])
  #   CUDA.@cuprintln("Ireduce: \t" , Ireduce[1], "\t", Ireduce[2], "\t", Ireduce[3])
  #   #CUDA.@cuprintln(iii[1], " ", iii[2], " ")
  #   #CUDA.@cuprintln(J[1], " ", J[2], " ")
  # end

  