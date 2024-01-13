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

SUM = false
PROD = false
MAX = false
MIN = false
UINT8 = false
UINT16 = false
UINT32 = false
UINT64 = false
UINT128 = false
INT8 = false
INT16 = false
INT32 = false
INT64 = false
INT128 = false
FLOAT16 = false
FLOAT32 = false

KA = true
#first arg needs to be CUDA or KA
for  ARG in ARGS
  if ARG == "KA" || ARG == "CUDA"
    global KA = ARG == "KA"
  elseif ARG == "SUM"
    global SUM = true
  elseif ARG == "PROD"
    global PROD = true
  elseif ARG == "MAX"
    global MAX = true
  elseif ARG == "MIN"
    global MIN = true
  elseif ARG == "UINT8"
    global UINT8 = true
  elseif ARG == "UINT16"
    global UINT16 = true
  elseif ARG == "UINT32"
    global UINT32 = true
  elseif ARG == "UINT64"
    global UINT64 = true
  elseif ARG == "UINT128"
    global UINT128 = true
  elseif ARG == "INT8"
    global INT8 = true
  elseif ARG == "INT16"
    global INT16 = true
  elseif ARG == "INT32"
    global INT32 = true
  elseif ARG == "INT64"
    global INT64 = true
  elseif ARG == "INT128"
    global INT128 = true
  elseif ARG == "FLOAT16"
    global FLOAT16 = true
  elseif ARG == "FLOAT32"
    global FLOAT32 = true
  end
end

if !SUM && !PROD && !MAX && !MIN
  global SUM = true
  global PROD = true
  global MAX = true
  global MIN = true
end

if !UINT8 && !UINT16 && !UINT32 && !UINT64 && !UINT128 && !INT8 && !INT16 && !INT32 && !INT64 && !INT128 && !FLOAT16 && !FLOAT32
  global UINT8 = true
  global UINT16 = true
  global UINT32 = true
  global UINT64 = true
  global UINT128 = true
  global INT8 = true
  global INT16 = true
  global INT32 = true
  global INT64 = true
  global INT128 = true
  global FLOAT16 = true
  global FLOAT32 = true
end

path = dirname(@__FILE__)

const file = joinpath(path, joinpath("CUDA_scalar.csv"))
const KAfile = joinpath(path, joinpath("KA_scalar_v1.csv"))

function benchmark_CUDA_scalar(inputType, op, init; write_header=false, warmup=false)
  n =128
  while n < 8000000
      results = []
      N = []
      types = []
      operators = []
  
      # this will take longer as every iteration the function will be parsed
      if KA
        bench = @benchmarkable CUDA.@sync(mapreducedim(x->x, $op, final, data; init=$init)) evals=10 samples=500 seconds = 10000 setup= (begin 
            data=CUDA.rand($inputType, $n)
            final=CUDA.ones($inputType, 1) 
          end) teardown = (begin 
            #CUDA.unsafe_free!(data) 
            #CUDA.unsafe_free!(final) 
          end)
      else
        bench = @benchmarkable CUDA.@sync(GPUArrays.mapreducedim!(x->x, $op, final, data; init=$init)) evals=10 samples=500 seconds = 10000 setup= (begin 
            data=CUDA.rand($inputType, $n)
            final=CUDA.ones($inputType, 1) 
          end) teardown = (begin 
            #CUDA.unsafe_free!(data) 
            #CUDA.unsafe_free!(final) 
          end)
      end

      result = run(bench)

      push!(results, result)
      push!(N, n)
      push!(types, inputType)
      push!(operators, op)

      n = n * 2

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
  for idk in 1:7

    # ########################################
    # Sum
    # ########################################
    if SUM
      if UINT8 benchmark_CUDA_scalar(UInt8, +, UInt8(0)) end
      if UINT16 benchmark_CUDA_scalar(UInt16, +, UInt16(0)) end
      if UINT32 benchmark_CUDA_scalar(UInt32, +, UInt32(0)) end
      if UINT64 benchmark_CUDA_scalar(UInt64, +, UInt64(0)) end
      if UINT128 benchmark_CUDA_scalar(UInt128, +, UInt128(0)) end

      if INT8 benchmark_CUDA_scalar(Int8, +, Int8(0)) end
      if INT16 benchmark_CUDA_scalar(Int16, +, Int16(0)) end
      if INT32 benchmark_CUDA_scalar(Int32, +, Int32(0)) end
      if INT64 benchmark_CUDA_scalar(Int64, +, Int64(0)) end
      if INT128 benchmark_CUDA_scalar(Int128, +, Int128(0)) end

      if FLOAT16 benchmark_CUDA_scalar(Float16, +, Float16(0)) end
      if FLOAT32 benchmark_CUDA_scalar(Float32, +, Float32(0)) end
    end

    # ########################################
    # Min
    # ########################################  
    if MIN
      if UINT8 benchmark_CUDA_scalar(UInt8, min, typemax(UInt8)) end
      if UINT16 benchmark_CUDA_scalar(UInt16, min, typemax(UInt16)) end
      if UINT32 benchmark_CUDA_scalar(UInt32, min, typemax(UInt32)) end
      if UINT64 benchmark_CUDA_scalar(UInt64, min, typemax(UInt64)) end
      if UINT128 benchmark_CUDA_scalar(UInt128, min, typemax(UInt128)) end

      if INT8 benchmark_CUDA_scalar(Int8, min, typemax(Int8)) end
      if INT16 benchmark_CUDA_scalar(Int16, min, typemax(Int16)) end
      if INT32 benchmark_CUDA_scalar(Int32, min, typemax(Int32)) end
      if INT64 benchmark_CUDA_scalar(Int64, min, typemax(Int64)) end
      if INT128 benchmark_CUDA_scalar(Int128, min, typemax(Int128)) end

      if FLOAT16 benchmark_CUDA_scalar(Float16, min, typemax(Float16)) end
      if FLOAT32 benchmark_CUDA_scalar(Float32, min, typemax(Float32)) end
    end

    # ########################################
    # Max
    # ######################################## 
    if MAX
      if UINT8 benchmark_CUDA_scalar(UInt8, max, typemin(UInt8)) end
      if UINT16 benchmark_CUDA_scalar(UInt16, max, typemin(UInt16)) end
      if UINT32 benchmark_CUDA_scalar(UInt32, max, typemin(UInt32)) end
      if UINT64 benchmark_CUDA_scalar(UInt64, max, typemin(UInt64)) end
      if UINT128 benchmark_CUDA_scalar(UInt128, max, typemin(UInt128)) end

      if INT8 benchmark_CUDA_scalar(Int8, max, typemin(Int8)) end
      if INT16 benchmark_CUDA_scalar(Int16, max, typemin(Int16)) end
      if INT32 benchmark_CUDA_scalar(Int32, max, typemin(Int32)) end
      if INT64 benchmark_CUDA_scalar(Int64, max, typemin(Int64)) end
      if INT128 benchmark_CUDA_scalar(Int128, max, typemin(Int128)) end

      if FLOAT16 benchmark_CUDA_scalar(Float16, max, typemin(Float16)) end
      if FLOAT32 benchmark_CUDA_scalar(Float32, max, typemin(Float32)) end
    end

    # ########################################
    # Product
    # ######################################## 
    if PROD
      if UINT8 benchmark_CUDA_scalar(UInt8, *, UInt8(1)) end
      if UINT16 benchmark_CUDA_scalar(UInt16, *, UInt16(1)) end
      if UINT32 benchmark_CUDA_scalar(UInt32, *, UInt32(1)) end
      if UINT64 benchmark_CUDA_scalar(UInt64, *, UInt64(1)) end
      if UINT128 benchmark_CUDA_scalar(UInt128, *, UInt128(1)) end

      if INT8 benchmark_CUDA_scalar(Int8, *, Int8(1)) end
      if INT16 benchmark_CUDA_scalar(Int16, *, Int16(1)) end
      if INT32 benchmark_CUDA_scalar(Int32, *, Int32(1)) end
      if INT64 benchmark_CUDA_scalar(Int64, *, Int64(1)) end
      if INT128 benchmark_CUDA_scalar(Int128, *, Int128(1)) end

      if FLOAT16 benchmark_CUDA_scalar(Float16, *, Float16(1)) end
      if FLOAT32 benchmark_CUDA_scalar(Float32, *, Float32(1)) end
    end
  end
end

benchmark_CUDA_scalar()