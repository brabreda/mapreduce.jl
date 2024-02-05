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

path = dirname(@__FILE__)

const file = path*"CUDA_ND_elements_per_group.csv"
const KAfile = path*"KA_ND_elements_per_group_v1.csv"
const KA = true


function benchmark_CUDA_ND_elements_per_group(inputType, op, init; write_header=false)
  for n in 2 .^ vcat(collect(7:22),collect(reverse(7:22)))
      results = []
      types = []
      operators = []
      N = []

      @show n

      inputdim = (n, 32)
      outputdim = (1, 32)

      data = CuArray{inputType}(undef, inputdim)
      final = CuArray{inputType}(undef, outputdim)

  
      # this will take longer as every iteration the function will be parsed
      if KA
        bench = @benchmarkable CUDA.@sync(mapreducedim(x->x, $op, $final, $data; init=$init)) evals=1 samples=1000 seconds = 10000 setup= (begin 
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
      push!(types, inputType)
      push!(operators, op)
      push!(N, n)

      df_benchmark = mapreduce(vcat, zip(results, N, types, operators)) do (x, y, a, b)
          df = DataFrame(x)
          df.N .= y
          df.type .= a
          df.op .= b
          df
      end
      
      if KA
        CSV.write(KAfile, df_benchmark;append=true, writeheader=write_header)

      else
        CSV.write(file, df_benchmark;append=true, writeheader=write_header)
      end      
      write_header = false
  end 
end

function benchmark_CUDA_ND_elements_per_group()

  write(KA ? KAfile : file, "times,gctimes,memory,allocs,N,type,op\n");
  for idk in 1:1
    println("\t\t", idk)

    # ########################################
    # Sum
    # ########################################
    # benchmark_CUDA_ND_elements_per_group(UInt8, +, UInt8(0))
    # benchmark_CUDA_ND_elements_per_group(UInt16, +, UInt16(0))
    # benchmark_CUDA_ND_elements_per_group(UInt32, +, UInt32(0))
    benchmark_CUDA_ND_elements_per_group(UInt64, +, UInt64(0))
    # benchmark_CUDA_ND_elements_per_group(UInt128, +, UInt128(0))

    # benchmark_CUDA_ND_elements_per_group(Int8, +, Int8(0))
    # benchmark_CUDA_ND_elements_per_group(Int16, +, Int16(0))
    # benchmark_CUDA_ND_elements_per_group(Int32, +, Int32(0))
    # benchmark_CUDA_ND_elements_per_group(Int64, +, Int64(0))
    # benchmark_CUDA_ND_elements_per_group(Int128, +, Int128(0))

    # benchmark_CUDA_ND_elements_per_group(Float16, +, Float16(0))
    # benchmark_CUDA_ND_elements_per_group(Float32, +, Float32(0))


    # ########################################
    # Min
    # ########################################  
    # benchmark_CUDA_ND_elements_per_group(UInt8, min, typemax(UInt8))
    # benchmark_CUDA_ND_elements_per_group(UInt16, min, typemax(UInt16))
    # benchmark_CUDA_ND_elements_per_group(UInt32, min, typemax(UInt32))
    # benchmark_CUDA_ND_elements_per_group(UInt64, min, typemax(UInt64))
    # benchmark_CUDA_ND_elements_per_group(UInt128, min, typemax(UInt128))

    # benchmark_CUDA_ND_elements_per_group(Int8, min, typemax(Int8))
    # benchmark_CUDA_ND_elements_per_group(Int16, min, typemax(Int16))
    # benchmark_CUDA_ND_elements_per_group(Int32, min, typemax(Int32))
    # benchmark_CUDA_ND_elements_per_group(Int64, min, typemax(Int64))
    # benchmark_CUDA_ND_elements_per_group(Int128, min, typemax(Int128))

    # benchmark_CUDA_ND_elements_per_group(Float16, min, typemax(Float16))
    # benchmark_CUDA_ND_elements_per_group(Float32, min, typemax(Float32))


    # ########################################
    # Min
    # ######################################## 
    # benchmark_CUDA_ND_elements_per_group(UInt8, max, typemin(UInt8))
    # benchmark_CUDA_ND_elements_per_group(UInt16, max, typemin(UInt16))
    # benchmark_CUDA_ND_elements_per_group(UInt32, max, typemin(UInt32))
    # benchmark_CUDA_ND_elements_per_group(UInt64, max, typemin(UInt64))
    # benchmark_CUDA_ND_elements_per_group(UInt128, max, typemin(UInt128))

    # benchmark_CUDA_ND_elements_per_group(Int8, max, typemin(Int8))
    # benchmark_CUDA_ND_elements_per_group(Int16, max, typemin(Int16))
    # benchmark_CUDA_ND_elements_per_group(Int32, max, typemin(Int32))
    # benchmark_CUDA_ND_elements_per_group(Int64, max, typemin(Int64))
    # benchmark_CUDA_ND_elements_per_group(Int128, max, typemin(Int128))

    # benchmark_CUDA_ND_elements_per_group(Float16, max, typemin(Float16))
    # benchmark_CUDA_ND_elements_per_group(Float32, max, typemin(Float32))


    # ########################################
    # Product
    # ######################################## 
    # benchmark_CUDA_ND_elements_per_group(UInt8, *, UInt8(1))
    # benchmark_CUDA_ND_elements_per_group(UInt16, *, UInt16(1))
    # benchmark_CUDA_ND_elements_per_group(UInt32, *, UInt32(1))
    # benchmark_CUDA_ND_elements_per_group(UInt64, *, UInt64(1))
    # benchmark_CUDA_ND_elements_per_group(UInt128, *, UInt128(1))

    # benchmark_CUDA_ND_elements_per_group(Int8, *, Int8(1))
    # benchmark_CUDA_ND_elements_per_group(Int16, *, Int16(1))
    # benchmark_CUDA_ND_elements_per_group(Int32, *, Int32(1))
    # benchmark_CUDA_ND_elements_per_group(Int64, *, Int64(1))
    # benchmark_CUDA_ND_elements_per_group(Int128, *, Int128(1))

    # benchmark_CUDA_ND_elements_per_group(Float16, *, Float16(1))
    # benchmark_CUDA_ND_elements_per_group(Float32, *, Float32(1))
  end
end

benchmark_CUDA_ND_elements_per_group()