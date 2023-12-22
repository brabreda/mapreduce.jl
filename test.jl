using BenchmarkTools
using Random
using CUDA

include("src/launch.jl")
include("src/CUDAlaunch.jl")
include("src/ndmapreduce.jl")



a1 = CUDA.rand(Float32, 1000000)
a = CUDA.ceil.(Int32, a1 * 5)
b = CUDA.zeros(Int32,1)

println("CUDA")
#display(@benchmark(CUDA.@sync(mapreduce(x->x, +, a, dims=(3,4)))))

#display(c)
#c = CUDA.@sync(mapreduce(x->x, +, a))
#display(c)


# a1 = CUDA.rand(Float32,100,100)
# a = CUDA.ceil.(Int32, a1 * 5)
# b = CUDA.zeros(Int32,1,1)

# println("CUDA")
# #display(@benchmark(CUDA.@sync(mapreduce(x->x, +, a, dims=(3,4)))))

CUDA.@sync(mapreducedim(x->x, +, b,a; init=Int32(0)))
x = CUDA.@profile CUDA.@sync(mapreducedim(x->x, +, b,a; init=Int32(0)))
y = CUDA.@profile CUDA.@sync(mapreducedim(x->x, +, b,a; init=Int32(0)))

display(x)
display(y)

# display(c)
# c = CUDA.@sync(mapreduce(x->x, +, a))
# display(c)



# println("=================================================== \n")
# println("KernelAbstractions To 1\n")
# println("=================================================== \n")

# #c = CUDA.@sync(mapreduce(x->x*2, +, a))

# #display(@benchmark(CUDA.@sync(mapreducedim(x->x, +, b,a; init=Float32(0.0)))))
# #c = @benchmark CUDA.@sync(mapreducedim(x->x*2, +, b,a; init=Int64(0)))

# #display(c)
# #display(c)
# #display(b)

# a1 = CUDA.rand(Float32,1024, 100)
# a = CUDA.ceil.(Int32, a1 * 5)
# b = CUDA.zeros(Int32,1,100)

# c = CUDA.@sync(mapreducedim(x->x, +, b,a; init=Int32(0.0)))
# display(c)
# c = CUDA.@sync(mapreduce(x->x, +, a, dims=(1)))
# display(c)

# println("=================================================== \n")
# println("KernelAbstractions To N\n")
# println("=================================================== \n")

# #d = @benchmark CUDA.@sync(mapreducedim(x->x*2, +, b,a; init=Int64(0)))




# display(c)
#display(d)
#display(b)

#abs.(c - d)


  