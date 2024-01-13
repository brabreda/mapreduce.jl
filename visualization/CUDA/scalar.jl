using CSV
using DataFrames
using Plots
using StatsPlots
using Plots.PlotMeasures

path = dirname(@__FILE__)

const CUDA_file =   joinpath(path, joinpath("../../benchmarks/CUDA/CUDA_scalar.csv"))
const KA_V1_file =  joinpath(path, joinpath("../../benchmarks/CUDA/KA_scalar_v1.csv"))
const CUB_file =    joinpath(path, joinpath("../../benchmarks/CUDA/CUB.csv"))

# get path reletave the current file
min_times_CUDA = DataFrame()
if isfile(CUDA_file) 
  CUDA_scalar = DataFrame(CSV.File(CUDA_file))
  if !isempty(CUDA_scalar)
    min_times_CUDA = combine(groupby(CUDA_scalar, [:N, :type, :op]), "times" => minimum => :min_time)
    min_times_CUDA[!, :impl] .= "CUDA.jl"
    min_times_CUDA[!, :name] .= "CUDA.jl "
  end
else
  @warn "CUDA_scalar.csv not found"
end

min_times_KA_V1 = DataFrame()
if isfile(KA_V1_file) 
  KA_V1_scalar = DataFrame(CSV.File(KA_V1_file))
  if !isempty(KA_V1_scalar)
    min_times_KA_V1 = combine(groupby(KA_V1_scalar, [:N, :type, :op]), "times" => minimum => :min_time)
    min_times_KA_V1[!, :name] .= "Vendor neutral 1 "
    min_times_KA_V1[!, :impl] .= "Vendor neutral 1"
  end
else
  @warn "KA_scalar_v1.csv not found"
end

min_times_KA_V3 = DataFrame()


min_times_CUB = DataFrame()
if isfile(CUB_file) 
  CUB = DataFrame(CSV.File(CUB_file))
  if !isempty(CUB)
    CUB = select(CUB, [:N,:sizetype,:type,:elapsed,:operation])

    #remane operaton to op and elapsed to times
    rename!(CUB, :operation => :op)
    rename!(CUB, :elapsed => :times)

    CUB[!, :type] = replace.(CUB[!, :type], "uint8_t" => "UInt8")
    CUB[!, :type] = replace.(CUB[!, :type], "uint16_t" => "UInt16")
    CUB[!, :type] = replace.(CUB[!, :type], "uint32_t" => "UInt32")
    CUB[!, :type] = replace.(CUB[!, :type], "uint64_t" => "UInt64")
    CUB[!, :type] = replace.(CUB[!, :type], "uint128_t" => "UInt128")

    CUB[!, :type] = replace.(CUB[!, :type], "int8_t" => "Int8")
    CUB[!, :type] = replace.(CUB[!, :type], "int16_t" => "Int16")
    CUB[!, :type] = replace.(CUB[!, :type], "int32_t" => "Int32")
    CUB[!, :type] = replace.(CUB[!, :type], "int64_t" => "Int64")
    CUB[!, :type] = replace.(CUB[!, :type], "int128_t" => "Int128")

    #float to float16 and double to float32
    CUB[!, :type] = replace.(CUB[!, :type], "float" => "Float16")
    CUB[!, :type] = replace.(CUB[!, :type], "double" => "Float32")

    min_times_CUB = combine(groupby(CUB, [:N, :type, :op]), "times" => minimum => :min_time)
    min_times_CUB[!, :name] .= "CUB "
    min_times_CUB[!, :impl] .= "CUB"
  end
else
  @warn "CUB.csv not found"
end



merged_df = vcat(min_times_CUDA, min_times_KA_V1)
merged_df = vcat(merged_df, min_times_KA_V3)

if !isempty(merged_df)
  merged_df[!, :min_time] = merged_df[!, :min_time] ./ 1000
end

merged_df = vcat(merged_df, min_times_CUB)

# filter N below 500000 to get figure ...
# merged_df = filter(row -> row[:N] <= 500000, merged_df)

#get unique types
if !isempty(merged_df)
  if !isdir(joinpath(path, "scalar"))
    mkdir(joinpath(path, "scalar"))
  end

  merged_df[!, :op] = replace.(merged_df[!, :op], "+" => "sum")

  merged_df[!, :op] = replace.(merged_df[!, :op], "*" => "prod")

  types = unique(merged_df[!, :type])
  #get unique operations
  ops = unique(merged_df[!, :op])

  #iteratore over each combination of type and operation
  for type in types
      for op in ops
          #filter the dataframe for the current type and operation
          plot(xaxis=:log2, title="Minimum execution time scalar "*op*" reduction "*type, xlabel="N", ylabel="Time (μs)", legend=:topleft, size=(800, 600), link=:both, left_margin = [10mm 0mm], bottom_margin = [10mm 0mm], right_margin = [10mm 0mm])

          filtered_df = filter(row -> row[:type] == type && row[:op] == op, merged_df)
          grouped_df = groupby(filtered_df, [:name, :impl, :type, :op])

          for group in grouped_df

          name, impl = group[1, :name], group[1, :impl]
          plot!(group.N, group.min_time, label="$name", marker=:circle, color= (impl == "CUDA.jl" ? :red : impl == "CUB" ? :blue : :green))
          end

          png(joinpath(path, joinpath("scalar/"*type*"_"*op*".png")))
      end
  end
end