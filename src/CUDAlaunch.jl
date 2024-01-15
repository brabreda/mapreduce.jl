# launch configuration
function launch_config(kernelObj::KernelAbstractions.Kernel{CUDABackend,K,L,M}, args...; workgroupsize, ndrange) where {K,L,M}
  dev = device()
  backend = KernelAbstractions.backend(kernelObj)
  ndrange, workgroupsize, iterspace, dynamic = KernelAbstractions.launch_config(kernelObj, ndrange ,workgroupsize)
  ctx = KernelAbstractions.mkcontext(kernelObj, ndrange, iterspace)

  kernel = CUDA.@cuda launch=false always_inline=backend.always_inline kernelObj.f(ctx, args...)

  maxblocks, maxthreads = CUDA.launch_configuration(kernel.fun)

  groupsize = if prod(workgroupsize) <= maxthreads
    CUDA.nextwarp(dev, prod(workgroupsize))
  else 
    CUDA.nextwarp(dev, cld(prod(workgroupsize), cld(prod(workgroupsize), maxthreads)))
  end

  ndrangesize = groupsize * maxblocks

  return groupsize, ndrangesize
end

function max_workgroupsize(::CUDABackend)
  dev = CUDA.device()
  return attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
end
