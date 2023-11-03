

mutable struct CudaLaunchConfig
  kernel
  ctx
  threads
  max_threads
  blocks
  CudaLaunchConfig(kernel, ctx, threads, max_threads, blocks) = new(kernel, ctx, threads, max_threads, blocks)
end

function (cfg::CudaLaunchConfig)(args...)
  cfg.kernel(cfg.ctx, args...; threads=cfg.threads, blocks=cfg.blocks)
end

function getStrideSize(cfg::CudaLaunchConfig)
  return cfg.threads
end

function getMaxGroups(cfg::CudaLaunchConfig)
  return cfg.blocks
end

function getMaxWorkItems(cfg::CudaLaunchConfig)
  return cfg.max_threads
end

function setWorkItems(cfg::CudaLaunchConfig, workitems)
  cfg.threads = workitems
end

# launch configuration
function launch_config(kernelObj, args...; workgroupsize, ndrange)
dev = device()
backend = KernelAbstractions.backend(kernelObj)
ndrange, workgroupsize, iterspace, dynamic = KernelAbstractions.launch_config(kernelObj, ndrange,workgroupsize)
ctx = KernelAbstractions.mkcontext(kernelObj, ndrange, iterspace)

kernel = CUDA.@cuda launch=false always_inline=backend.always_inline kernelObj.f(ctx, args...)

# we should tailor the amount of threads to the amount of work we have to do, a good measure
# of work is the groupsize. If the groupsize is smaller than the maximum hardware groupsize
# we can use less threads, if the groupssize is bigger we would equaly devide the work over 
# the threads (e.g. use less threads but with equal work per thread).
maxblocks, maxthreads = CUDA.launch_configuration(kernel.fun; max_threads=1024)

threads = if prod(workgroupsize) <= maxthreads
  CUDA.nextwarp(dev, prod(workgroupsize))
else
  CUDA.nextwarp(dev, cld(prod(workgroupsize),cld(prod(workgroupsize), maxthreads)))
end


return CudaLaunchConfig(kernel, ctx, threads, maxthreads, maxblocks)
end
