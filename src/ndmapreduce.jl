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
@kernel function partial_mapreduce_grid(f, op, neutral, strideSize, R, As...)
  
  # load the neutral value
  Iout = CartesianIndex(Tuple(@index(Group, Cartesian))..., 1)
  @inbounds neutral = if neutral === nothing
      R[Iout]
  else
      neutral
  end

  val = op(neutral, neutral)

  # reduce serially across chunks of input vector that don't fit in a block
  ireduce = @index(Local, Linear)
  @inbounds while ireduce <= prod(@groupsize())
      Ireduce = @inbounds KernelAbstractions.workitems( KernelAbstractions.__iterspace(__ctx__))[ireduce]
      J = max(@index(Group, Cartesian), Ireduce)
      val = op(val, f(_map_getindex(As, J)...))
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

  # iteration domain, one part covers the reductions that every group needs to reduce,
  # another part covers in what direction the groups should reduce
  workgroupDim = ifelse.(size(A) .== size(R), 1, size(A))
  ndrangeDim = size(A)

  # allocate an additional, empty dimension to write the reduced value to.
  # this does not affect the actual location in memory of the final values,
  # but allows us to write a generalized kernel supporting partial reductions.
  R′ = reshape(R, (size(R)..., 1))

  args = (f, op, init, 1, R′, A)
  kernelObj = partial_mapreduce_grid(KABackend)
  kernelLaunchConfig = launch_config(kernelObj, args...; workgroupsize=workgroupDim, ndrange=ndrangeDim)

  # the groupsize of the kernel is disatched form the hardware groupsize, this allows a group to process more
  # elements that the hardware groupsize would allow. A stride a how much elements our group can process in
  # parrallel.
  strideSize = getStrideSize(kernelLaunchConfig)
  maxgroups  = getMaxGroups(kernelLaunchConfig) # how many groups can be used for full occupancy

  other_groups = prod(size(R))
  reduce_groups = if strideSize <= other_groups
      1                                        # We don't need multiple groups to cover 1 slice.
  else
      min(cld(prod(workgroupDim), strideSize), # it can be optimal to use multiple groups for one slice
          cld(maxgroups, other_groups))        # but we should not use more groups than we have. We should
  end                                          # only do this out of necessity
  
  # The total groups we want are the groups needed to cover one slice times the number of slices.
  total_groups = reduce_groups * other_groups

  kernelLaunchConfig.blocks = total_groups

  # If we have only one group per slice, every slice can be reduced in one go, no second kernel is needed.
  if reduce_groups == 1
      kernelLaunchConfig( f, op, init, strideSize, R′, A)
  else
      # we need temporary storage to hold partial reductions for every slice the endresult can be calcultated
      # by reducing the temporary storage in the direction of the every slice.
      partial = similar(R, (size(R)..., ngroups))
      if init === nothing
          # without an explicit initializer we need to copy from the output container
          partial .= R
      end
      # NOTE: we can't use the previously-compiled kernel, since the type of `partial`
      #       might not match the original output container (e.g. if that was a view).
      kernelLaunchConfig( f, op, init, strideSize, partial, A)

      mapreducedim(identity, op, R′, partial; init=init)
  end

  return R
end