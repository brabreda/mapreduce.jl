using KernelAbstractions

"""
  launch_config(kernelObj, args...; workgroupsize, ndrange)

Return the maximum amout of threads and blocks giving the best occupancy for a kernel
- `kernelObj`: A KernelAbstractions kernel object 
- `args...`: Arguments for the kernel
- `workgroupsize`: Ideal groupsize without taking limits into account
- `ndrange`: Ideal ndrange without taking limits into account
"""
function launch_config(kernelObj, args...; workgroupsize, ndrange) end

"""
max_workgroupsize(backend)

Return the maximum amount of threads per group
- `kernel`: A KernelAbstractions backend 
"""
function max_workgroupsize(backend) end

"""
  max_localmemory(backend)

Return the maximum amount of local memory available per group
- `kernel`: A KernelAbstractions backend 
"""
function max_localmemory(backend) end




