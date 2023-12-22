using BenchmarkTools
using Random
using CUDA
using GPUArrays
using Profile
using PProf
using StatProfilerHTML

include("src/launch.jl")
include("src/CUDAlaunch.jl")
include("src/ndmapreduce.jl")

N = 2^10

# a1 = CUDA.rand(Float32,1000000)
# a = CUDA.ceil.(UInt32, a1 * 5)
# b = CUDA.zeros(UInt32,1)

#println("CUDA")
#display(@benchmark(CUDA.@sync(mapreduce(x->x, +, a, dims=(3,4)))))

println("=================================================== \n")
println("KernelAbstractions To 1\n")
println("=================================================== \n")

# display( CUDA.@sync(mapreducedim(x->x, +, b,a; init=UInt32(0))))
# display( CUDA.@sync(GPUArrays.mapreducedim!(x->x, +,b, a; init=UInt32(0))))


println("=================================================== \n")
println("CUDA To 1\n")
println("=================================================== \n")


#display(@benchmark(CUDA.@sync(mapreducedim(x->x, +, b,a; init=Float32(0.0)))))

#display(c)
#display(c)
#display(d)


a = CUDA.rand(Int32, (1024,32))
b = CUDA.rand(Int32, (1,32))

# println("=================================================== \n")
# println("KernelAbstractions To N\n")
# println("=================================================== \n")
#Profile.init(0x0000000000989680, 0.0001)

x = @btime CUDA.@sync(mapreduce(x->x, +, a,dims=(1); init=Int32(0)))
y = @btime CUDA.@sync(mapreducedim(x->x, +,b, a; init=Int32(0)))

display(x)
display(y)


a = CUDA.rand(Int32, (1024,100))
b = CUDA.rand(Int32, (1, 100))

# println("=================================================== \n")
# println("KernelAbstractions To N\n")
# println("=================================================== \n")
#Profile.init(0x0000000000989680, 0.0001)

# x = @btime CUDA.@sync(mapreducedim(x->x, +,b, a; init=Int32(0)))
# y = @btime CUDA.@sync(mapreduce(x->x, +, a,dims=(1); init=Int32(0)))

#display(x)
#display(y)

#@profilehtml CUDA.@sync(mapreducedim(x->x, +,b, a; init=Int32(0)))
#

# display(c)
# display(d)
#display(b)

#abs.(c - d)
#prof()



# indice = CartesianIndices((Base.OneTo(5), Base.OneTo(5)))
# a = rand(5, 5)

# function fooA(indice, A)
#     return A[indice[1]]
# end

# function fooB(indice, A...)
#   return _map_getindex(A,indice[1])
# end

# @benchmark fooA($indice, $a)

# @benchmark fooB($indice, $a)