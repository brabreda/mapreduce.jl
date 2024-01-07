# launch configuration
NVTX.@annotate function launch_config(kernelObj, args...; workgroupsize, ndrange)
  backend = KernelAbstractions.backend(kernelObj)
  ndrange, workgroupsize, iterspace, dynamic = KernelAbstractions.launch_config(kernelObj, ndrange ,workgroupsize)
  ctx = KernelAbstractions.mkcontext(kernelObj, ndrange, iterspace)
  
  NVTX.@range "@cuda launch=false" begin
  kernel = CUDA.@cuda launch=false always_inline=backend.always_inline kernelObj.f(ctx, args...)
  end

  maxblocks, maxthreads = CUDA.launch_configuration(kernel.fun)

  groupsize = if prod(workgroupsize) <= maxthreads
    maxthreads
  else 
    cld(prod(workgroupsize), cld(prod(workgroupsize), maxthreads))
  end

  ndrangesize = groupsize * maxblocks

  return groupsize, ndrangesize
end
