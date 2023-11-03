using BenchmarkTools
using Random
using CUDA

include("src/launch.jl")
include("src/launchconfig.jl")
include("src/ndmapreduce.jl")


a1 = CUDA.rand(Float32,35,35,35,35)
a = CUDA.ceil.(Int32, a1 * 5)
b = CUDA.zeros(Int64,35,35,1,1)

println("CUDA")
#display(@benchmark(CUDA.@sync(mapreduce(x->x, +, a, dims=(3,4)))))
c = CUDA.@sync(mapreduce(x->x*2, +, a, dims=(3,4)))

println("KernelAbstractions")
#display(@benchmark(CUDA.@sync(mapreducedim(x->x, +, b,a; init=Float32(0.0)))))
d = CUDA.@sync(mapreducedim(x->x*2, +, b,a; init=Int64(0)))

#display(a)
#display(c)
#display(d)

#abs.(c - d)


  