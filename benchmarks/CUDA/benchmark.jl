using BenchmarkTools
using Random
using CUDA
using GPUArrays
using DataFrames
using CSV
using NVTX

include("../benchmarks_to_df.jl")
include("../../src/ndmapreduce.jl")
include("../../src/launch.jl")
include("../../src/CUDAlaunch.jl")


const file = "CUDA_scalar.csv"
const KAfile = "KA_scalar_v1.csv"
const KA = true

function benchmark_CUDA_scalar(inputType, op, init; write_header=false, warmup=false)
  for n in 2 .^ vcat(collect(reverse(22:28)),collect(22:28))
     results = []
     N = []
     types = []
     operators = []
     sleep(0.5)

     @show n

     device_synchronize()
     GC.gc(true)

     CUDA.reclaim()
     #CUDA.memory_status()

     data = CuArray{inputType}(undef, n)
     final = CuArray{inputType}(undef, 1)

     # this will take longer as every iteration the function will be parsed
     if KA
       bench = @benchmarkable CUDA.@sync(mapreducedim(x->x, $op, $final, $data; init=$init)) evals=1 samples=1000 seconds = 10000 setup= (begin 
         device_synchronize()
         rand!($data)
         rand!($final)
         device_synchronize()
       end)
     else
       bench = @benchmarkable CUDA.@sync(GPUArrays.mapreducedim!(x->x, $op, $final, $data; init=$init)) evals=1 samples=1000 seconds = 10000 setup= (begin 
           device_synchronize()
           rand!($data)
           rand!($final)
           device_synchronize()
         end)
     end
     result = run(bench)

     CUDA.unsafe_free!(data) 
     CUDA.unsafe_free!(final)

     push!(results, result[2:end])
     push!(N, n)
     push!(types, inputType)
     push!(operators, op)

     df_benchmark = mapreduce(vcat, zip(results, N, types, operators)) do (x, y, a, b)
         df = DataFrame(x)
         df.N .= y
         df.type .= a
         df.op .= b
         df
     end
     if !warmup
   
       if KA
         CSV.write(KAfile, df_benchmark;append=true, writeheader=write_header)
       else
         CSV.write(file, df_benchmark;append=true, writeheader=write_header)
       end
       write_header = false
     end
 end 
end

function benchmark_CUDA_scalar()

  write(KA ? KAfile : file, "times,gctimes,memory,allocs,N,type,op\n");
  for idk in 1:1
  # ########################################
  # Sum
  # ########################################
  # benchmark_CUDA_scalar(UInt8, +, UInt8(0))
  # benchmark_CUDA_scalar(UInt16, +, UInt16(0))
  # benchmark_CUDA_scalar(UInt32, +, UInt32(0))
  benchmark_CUDA_scalar(UInt64, +, UInt64(0))
  # benchmark_CUDA_scalar(UInt128, +, UInt128(0))

  # benchmark_CUDA_scalar(Int8, +, Int8(0))
  # benchmark_CUDA_scalar(Int16, +, Int16(0))
  # benchmark_CUDA_scalar(Int32, +, Int32(0))
  # benchmark_CUDA_scalar(Int64, +, Int64(0))
  # benchmark_CUDA_scalar(Int128, +, Int128(0))

  # benchmark_CUDA_scalar(Float16, +, Float16(0))
  # benchmark_CUDA_scalar(Float32, +, Float32(0))


  # # ########################################
  # # Min
  # # ########################################  
  # benchmark_CUDA_scalar(UInt8, min, typemax(UInt8))
  # benchmark_CUDA_scalar(UInt16, min, typemax(UInt16))
  # benchmark_CUDA_scalar(UInt32, min, typemax(UInt32))
  # benchmark_CUDA_scalar(UInt64, min, typemax(UInt64))
  # benchmark_CUDA_scalar(UInt128, min, typemax(UInt128))

  # benchmark_CUDA_scalar(Int8, min, typemax(Int8))
  # benchmark_CUDA_scalar(Int16, min, typemax(Int16))
  # benchmark_CUDA_scalar(Int32, min, typemax(Int32))
  # benchmark_CUDA_scalar(Int64, min, typemax(Int64))
  # benchmark_CUDA_scalar(Int128, min, typemax(Int128))

  # benchmark_CUDA_scalar(Float16, min, typemax(Float16))
  # benchmark_CUDA_scalar(Float32, min, typemax(Float32))


  # # ########################################
  # # Min
  # # ######################################## 
  # benchmark_CUDA_scalar(UInt8, max, typemin(UInt8))
  # benchmark_CUDA_scalar(UInt16, max, typemin(UInt16))
  # benchmark_CUDA_scalar(UInt32, max, typemin(UInt32))
  # benchmark_CUDA_scalar(UInt64, max, typemin(UInt64))
  # benchmark_CUDA_scalar(UInt128, max, typemin(UInt128))

  # benchmark_CUDA_scalar(Int8, max, typemin(Int8))
  # benchmark_CUDA_scalar(Int16, max, typemin(Int16))
  # benchmark_CUDA_scalar(Int32, max, typemin(Int32))
  # benchmark_CUDA_scalar(Int64, max, typemin(Int64))
  # benchmark_CUDA_scalar(Int128, max, typemin(Int128))

  # benchmark_CUDA_scalar(Float16, max, typemin(Float16))
  # benchmark_CUDA_scalar(Float32, max, typemin(Float32))


  # # ########################################
  # # Product
  # # ######################################## 
  # benchmark_CUDA_scalar(UInt8, *, UInt8(1))
  # benchmark_CUDA_scalar(UInt16, *, UInt16(1))
  # benchmark_CUDA_scalar(UInt32, *, UInt32(1))
  # benchmark_CUDA_scalar(UInt64, *, UInt64(1))
  # benchmark_CUDA_scalar(UInt128, *, UInt128(1))

  # benchmark_CUDA_scalar(Int8, *, Int8(1))
  # benchmark_CUDA_scalar(Int16, *, Int16(1))
  # benchmark_CUDA_scalar(Int32, *, Int32(1))
  # benchmark_CUDA_scalar(Int64, *, Int64(1))
  # benchmark_CUDA_scalar(Int128, *, Int128(1))

  # benchmark_CUDA_scalar(Float16, *, Float16(1))
  # benchmark_CUDA_scalar(Float32, *, Float32(1))

  end
end

benchmark_CUDA_scalar()