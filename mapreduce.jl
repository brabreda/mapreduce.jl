module mapreduce

import KernelAbstractions 
import GPUArrays

@kernel function reduce_kernel(f, op, neutral, R, A , conf)
    # values for the kernel
    threadIdx_local = @index(Local)
    threadIdx_global = @index(Global)
    groupIdx = @index(Group)
    gridsize = @ndrange()[1]

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
    val = @groupreduce(op, val, neutral, conf)

    # use helper function to deal with atomic/non atomic reductions
    if threadIdx_local == 1
        if conf.use_atomics
            # this won't work
            KernelAbstractions.@atomic R[1] = op(R[1], val)
        else
            @inbounds R[groupIdx] = val
        end
    end
end

function GPUArrays.mapreducedim!(f::F, op::OP, R, A::Union{AbstractArray,Broadcast.Broadcasted};
                                                                                init=nothing) where {F, OP, T}
Base.check_reducedims(R, A)
length(A) == 0 && return R

if ndims(R) < ndims(A)
dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
R = reshape(R, dims)
end

Rall = CartesianIndices(axes(A))
Rother = CartesianIndices(axes(R))
Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))

R′ = reshape(R, (size(R)..., 1))

wanted_threads = shuffle ? nextwarp(dev, length(Rreduce)) : length(Rreduce)
function compute_threads(max_threads)
    if wanted_threads > max_threads
        shuffle ? prevwarp(dev, max_threads) : max_threads
    else
        wanted_threads
    end
end

# how many threads can we launch?
#
# we might not be able to launch all those threads to reduce each slice in one go.
# that's why each threads also loops across their inputs, processing multiple values
# so that we can span the entire reduction dimension using a single thread block.
kernel = @cuda launch=false partial_mapreduce_grid(f, op, init, Rreduce, Rother, Val(shuffle), R′, A)
compute_shmem(threads) = shuffle ? 0 : threads*sizeof(T)
kernel_config = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
reduce_threads = compute_threads(kernel_config.threads)
reduce_shmem = compute_shmem(reduce_threads)

# how many blocks should we launch?
#
# even though we can always reduce each slice in a single thread block, that may not be
# optimal as it might not saturate the GPU. we already launch some blocks to process
# independent dimensions in parallel; pad that number to ensure full occupancy.
other_blocks = length(Rother)
reduce_blocks = if other_blocks >= kernel_config.blocks
1
else
min(cld(length(Rreduce), reduce_threads),       # how many we need at most
cld(kernel_config.blocks, other_blocks))    # maximize occupancy
end

# determine the launch configuration
threads = reduce_threads
shmem = reduce_shmem
blocks = reduce_blocks*other_blocks

    # perform the actual reduction
    if reduce_blocks == 1
    # we can cover the dimensions to reduce using a single block
        kernel(f, op, init, Rreduce, Rother, Val(shuffle), R′, A; threads, blocks, shmem)
    else
        # we need multiple steps to cover all values to reduce
        partial = similar(R, (size(R)..., reduce_blocks))
        if init === nothing
            # without an explicit initializer we need to copy from the output container
            partial .= R
        end
        # NOTE: we can't use the previously-compiled kernel, since the type of `partial`
        #       might not match the original output container (e.g. if that was a view).
        @cuda(threads=threads, blocks=blocks, shmem=shmem,
        partial_mapreduce_grid(f, op, init, Rreduce, Rother, Val(shuffle), partial, A))

        GPUArrays.mapreducedim!(identity, op, R′, partial; init=init)
    end

    return R
end


# function GPUArrays.mapreducedim!(f::F, op::OP, R, A::Union{AbstractArray,Broadcast.Broadcasted}; init=nothing) where {F, OP}

#     Base.check_reducedims(R, A)
#     length(A) == 0 && return R # isempty(::Broadcasted) iterates

#     # add singleton dimensions to the output container, if needed
#     if ndims(R) < ndims(A)
#         dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
#         R = reshape(R, dims)
#     end
  
#     backend = KernelAbstractions.get_backend(A) 

#     conf = if conf == nothing get_reduce_config(backend, op, eltype(A)) else conf end
#     if length(R) == 1
#         if length(A) <= conf.items_per_workitem * conf.groupsize

#             reduce_kernel(backend, conf.groupsize)(f, op, init, R, A, conf, ndrange=conf.groupsize)
#             return R
#         else
#             # How many workitems do we want?
#             gridsize = cld(length(A), conf.items_per_workitem)
#             # how many workitems can we have?
#             gridsize = min(gridsize, conf.max_ndrange)

#             groups = cld(gridsize, conf.groupsize)
#             partial = conf.use_atomics==true ? R : similar(R, (size(R)...,groups))

#             reduce_kernel(backend, conf.groupsize)(f, op, init, partial, A, conf, ndrange=gridsize)
            
#             if !conf.use_atomics
#                 # correct this
#                 #__devicereduce(x->x, op, R, partial,init,conf, backend,Val(1))
#                 return R
#             end

#             return R
#         end
#     else
#         Rall = CartesianIndices(axes(A))
#         Rother = CartesianIndices(axes(R))
#         Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))

#         R′ = reshape(R, (size(R)..., 1))

#         ndrange, workgroupsize, iterspace, dynamic = KernelAbstractions.launch_config(kernel, ndrange, workgroupsize)

#     end
# end

end