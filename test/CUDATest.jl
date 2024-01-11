using CUDA

include("../src/ndmapreduce.jl")
include("../src/launch.jl")
include("../src/CUDAlaunch.jl")
include("base.jl")

eltypes = ( Int16, Int32, Int64,
            Float16, Float32, Float64,
            ComplexF16, ComplexF32, # ComplexF64, does not work to much rescources
            Complex{Int16}, Complex{Int32} # ,Complex{Int64} does not work to much rescources 
            )

reductionTest(CuArray, eltypes)