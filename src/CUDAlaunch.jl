# launch configuration
function launch_config(kernelObj::KernelAbstractions.Kernel{CUDABackend,K,L,M}, args...;localmeory::Union{Int}=0, workgroupsize, ndrange) where {K,L,M}
  dev = device()
  backend = KernelAbstractions.backend(kernelObj)
  ndrange, workgroupsize, iterspace, dynamic = KernelAbstractions.launch_config(kernelObj, ndrange ,workgroupsize)
  ctx = KernelAbstractions.mkcontext(kernelObj, ndrange, iterspace)

  kernel = CUDA.@cuda launch=false always_inline=backend.always_inline kernelObj.f(ctx, args...)

  maxblocks, maxthreads = CUDA.launch_configuration(kernel.fun)

  # allow user to take control of maxthreads based on max localmemory
  # if localmemory isa Calleble
  #   maxthreads = localmemory(maxthreads)
  # end

  groupsize = if prod(workgroupsize) <= maxthreads
    maxthreads
  else 
    CUDA.nextwarp(dev, cld(prod(workgroupsize), cld(prod(workgroupsize), maxthreads)))
  end

  ndrangesize = groupsize * maxblocks

  return groupsize, ndrangesize
end
