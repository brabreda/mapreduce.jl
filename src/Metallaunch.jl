# launch configuration
function launch_config(kernelObj::KernelAbstractions.Kernel{MetalBackend,K,L,M}, args...; workgroupsize, ndrange) where {K,L,M}
  ndrange, workgroupsize, iterspace, dynamic = KernelAbstractions.launch_config(kernelObj, ndrange ,workgroupsize)
  ctx = KernelAbstractions.mkcontext(kernelObj, ndrange, iterspace)

  kernel = @metal launch=false kernelObj.f(ctx, args...)

  # The pipeline state automatically computes occupancy stats
  maxthreads = Int(kernel.pipeline.maxTotalThreadsPerThreadgroup)
  maxblocks  = cld(prod(ndrange), maxthreads) * maxthreads

  ndrangesize = maxthreads * maxblocks

  return maxthreads, ndrangesize
end

function max_workgroupsize(::MetalBackend)
  dev = current_device()
  return Int(dev.maxThreadsPerThreadgroup.width)
end