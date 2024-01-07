using BenchmarkTools
using Random
using CUDA
using Profile
using PProf
using GPUArrays
using NVTX

include("src/launch.jl")
include("src/CUDAlaunch.jl")
include("src/ndmapreduce.jl")


# N = 2^8

# @show N

# a = CUDA.rand(UInt64, (N))
# b = CUDA.rand(UInt64, (1))

# # CUDA.@sync(mapreduce(x->x, +, a; init=UInt64(0)))
# # Profile.init(0x0000000059989680, 0.0000001)
# # Profile.clear()

# bench = @benchmarkable CUDA.@sync(mapreducedim(x->x, +, final, data; init=UInt64(0))) evals=10 samples=500 seconds = 10000 setup= (begin 
#             data=CUDA.rand(UInt64, ($N))
#             final=CUDA.ones(UInt64, (1)) 
#           end)
# run(bench)

# bench =  @benchmarkable CUDA.@sync(GPUArrays.mapreducedim!(x->x, +, final, data; init=UInt64(0))) evals=10 samples=500 seconds = 10000 setup= (begin 
#           data=CUDA.rand(UInt64, ($N))
#           final=CUDA.ones(UInt64, (1)) 
#         end)
# run(bench)

N = 2^15


data=CUDA.rand(UInt64, (N,32))
final=CUDA.ones(UInt64, (1,32)) 

 CUDA.@sync(mapreducedim(x->x, +, final, data; init=UInt64(0)))

# NVTX.enable_gc_hooks()

# bench = @benchmarkable begin NVTX.@range "mapreduce KA" begin CUDA.@sync(mapreducedim(x->x, +, final, data; init=UInt64(0))) end end evals=10 samples=500 seconds = 10000 setup= (begin 
#             data=CUDA.rand(UInt64, ($N))
#             final=CUDA.ones(UInt64, (1)) 
#           end)
# result = run(bench)
# display(result)

# bench = @benchmarkable begin NVTX.@range "mapreduce" begin  CUDA.@sync(GPUArrays.mapreducedim!(x->x, +, final, data; init=UInt64(0))) end end evals=10 samples=500 seconds = 10000 setup= (begin 
#           data=CUDA.rand(UInt64, ($N))
#           final=CUDA.ones(UInt64, (1)) 
#         end)
# result = run(bench)
# display(result)


# Profile.init(0x0000000059989680, 0.0000001)
# Profile.clear()









#Profile.clear()
#@profile CUDA.@sync(mapreducedim(x->x, +, b,a; init=UInt64(0)))

#ProfileSVG.save(joinpath("prof",("$N.svg")),timeunit=:ticks)




#println("CUDA")
#display(@benchmark(CUDA.@sync(mapreduce(x->x, +, a, dims=(3,4)))))

#display(c)
#c = CUDA.@sync(mapreduce(x->x, +, a))
#display(c)


# a1 = CUDA.rand(Float32,100,100)
# a = CUDA.ceil.(Int32, a1 * 5)
# b = CUDA.zeros(Int32,1,1)

# println("CUDA")
# #display(@benchmark(CUDA.@sync(mapreduce(x->x, +, a, dims=(3,4)))))

# CUDA.@sync(mapreducedim(x->x, +, b,a; init=Int32(0)))
# x = CUDA.@profile CUDA.@sync(mapreducedim(x->x, +, b,a; init=Int32(0)))
# y = CUDA.@profile CUDA.@sync(mapreducedim(x->x, +, b,a; init=Int32(0)))

# display(x)
# display(y)



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


  