using Metal

include("../src/ndmapreduce.jl")
include("../src/launch.jl")
include("../src/Metallaunch.jl")
include("base.jl")

eltypes = ( Int16, Int32, Int64,
            Float16, Float32,
            ComplexF16, ComplexF32,
            Complex{Int16}, Complex{Int32}, 
            Complex{Int64})


reductionTest(MtlArray, eltypes)