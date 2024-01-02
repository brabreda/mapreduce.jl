using BenchmarkTools
using Random
using CUDA
using GPUArrays
using DataFrames
using CSV

include("../benchmarks_to_df.jl")
include("../../src/ndmapreduce.jl")
include("../../src/launch.jl")
include("../../src/CUDAlaunch.jl")

path = dirname(@__FILE__)

const file = path*"CUDA_ND_dims.csv"
const KAfile = path*"KA_ND_dims_v1.csv"
const KA = true


function benchmark_CUDA_ND_dims(inputType, op, init; write_header=false)
  println(inputType)
  n =128 
  inputdims = (1024,2)
  outputdim = (1, 2)
  while prod(inputdim) < 4000000
      results = []
      types = []
      operators = []
      N = []


      println("\t", n)
      inputdim = (inputdims..., 2)
      outputdim = (outputdims..., 2)


  
      # this will take longer as every iteration the function will be parsed
      # if KA
      #   bench = @benchmarkable CUDA.@sync(mapreducedim(x->x, $op, final, data; init=$init)) evals=10 samples=500 seconds = 10000 setup= (begin 
      #     data=CUDA.rand($inputType, $inputdim)
      #     final=CUDA.rand($inputType, $outputdim) 
      #   end) teardown = (begin 
      #     CUDA.unsafe_free!(data) 
      #     CUDA.unsafe_free!(final) 
      #   end)
      # else
      #   bench = @benchmarkable CUDA.@sync(GPUArrays.mapreducedim!(x->x, $op, final, data; init=$init)) evals=10 samples=500 seconds = 10000 setup= (begin 
      #     data=CUDA.rand($inputType, $inputdim)
      #     final=CUDA.rand($inputType, $outputdim)  
      #   end) teardown = (begin 
      #     CUDA.unsafe_free!(data) 
      #     CUDA.unsafe_free!(final) 
      #   end)
      # end
      result = run(bench)

      push!(results, result)
      push!(types, inputType)
      push!(operators, op)
      push!(N, length(inputdim))

      n = n * 2

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

function benchmark_CUDA_ND_dims()

  write(KA ? KAfile : file, "times,gctimes,memory,allocs,N,type,op\n");
  for idk in 1:7
    println("\t\t", idk)

    # ########################################
    # Sum
    # ########################################
    benchmark_CUDA_ND_dims(UInt8, +, UInt8(0))
    benchmark_CUDA_ND_dims(UInt16, +, UInt16(0))
    benchmark_CUDA_ND_dims(UInt32, +, UInt32(0))
    benchmark_CUDA_ND_dims(UInt64, +, UInt64(0))
    benchmark_CUDA_ND_dims(UInt128, +, UInt128(0))

    benchmark_CUDA_ND_dims(Int8, +, Int8(0))
    benchmark_CUDA_ND_dims(Int16, +, Int16(0))
    benchmark_CUDA_ND_dims(Int32, +, Int32(0))
    benchmark_CUDA_ND_dims(Int64, +, Int64(0))
    benchmark_CUDA_ND_dims(Int128, +, Int128(0))

    benchmark_CUDA_ND_dims(Float16, +, Float16(0))
    benchmark_CUDA_ND_dims(Float32, +, Float32(0))


    # ########################################
    # Min
    # ########################################  
    benchmark_CUDA_ND_dims(UInt8, min, typemax(UInt8))
    benchmark_CUDA_ND_dims(UInt16, min, typemax(UInt16))
    benchmark_CUDA_ND_dims(UInt32, min, typemax(UInt32))
    benchmark_CUDA_ND_dims(UInt64, min, typemax(UInt64))
    benchmark_CUDA_ND_dims(UInt128, min, typemax(UInt128))

    benchmark_CUDA_ND_dims(Int8, min, typemax(Int8))
    benchmark_CUDA_ND_dims(Int16, min, typemax(Int16))
    benchmark_CUDA_ND_dims(Int32, min, typemax(Int32))
    benchmark_CUDA_ND_dims(Int64, min, typemax(Int64))
    benchmark_CUDA_ND_dims(Int128, min, typemax(Int128))

    benchmark_CUDA_ND_dims(Float16, min, typemax(Float16))
    benchmark_CUDA_ND_dims(Float32, min, typemax(Float32))


    # ########################################
    # Min
    # ######################################## 
    benchmark_CUDA_ND_dims(UInt8, max, typemin(UInt8))
    benchmark_CUDA_ND_dims(UInt16, max, typemin(UInt16))
    benchmark_CUDA_ND_dims(UInt32, max, typemin(UInt32))
    benchmark_CUDA_ND_dims(UInt64, max, typemin(UInt64))
    benchmark_CUDA_ND_dims(UInt128, max, typemin(UInt128))

    benchmark_CUDA_ND_dims(Int8, max, typemin(Int8))
    benchmark_CUDA_ND_dims(Int16, max, typemin(Int16))
    benchmark_CUDA_ND_dims(Int32, max, typemin(Int32))
    benchmark_CUDA_ND_dims(Int64, max, typemin(Int64))
    benchmark_CUDA_ND_dims(Int128, max, typemin(Int128))

    benchmark_CUDA_ND_dims(Float16, max, typemin(Float16))
    benchmark_CUDA_ND_dims(Float32, max, typemin(Float32))


    # ########################################
    # Product
    # ######################################## 
    benchmark_CUDA_ND_dims(UInt8, *, UInt8(1))
    benchmark_CUDA_ND_dims(UInt16, *, UInt16(1))
    benchmark_CUDA_ND_dims(UInt32, *, UInt32(1))
    benchmark_CUDA_ND_dims(UInt64, *, UInt64(1))
    benchmark_CUDA_ND_dims(UInt128, *, UInt128(1))

    benchmark_CUDA_ND_dims(Int8, *, Int8(1))
    benchmark_CUDA_ND_dims(Int16, *, Int16(1))
    benchmark_CUDA_ND_dims(Int32, *, Int32(1))
    benchmark_CUDA_ND_dims(Int64, *, Int64(1))
    benchmark_CUDA_ND_dims(Int128, *, Int128(1))

    benchmark_CUDA_ND_dims(Float16, *, Float16(1))
    benchmark_CUDA_ND_dims(Float32, *, Float32(1))
  end
end

benchmark_CUDA_ND_dims()