using CSV
using DataFrames
using Plots
using StatsPlots
using Plots.PlotMeasures

path = dirname(@__FILE__)

const Metal_file =   joinpath(path, joinpath("../../benchmarks/Metal/Metal_scalar.csv"))
const KA_V1_file =  joinpath(path, joinpath("../../benchmarks/Metal/KA_scalar_v1.csv"))
const KA_V3_file =  joinpath(path, joinpath("../../benchmarks/Metal/KA_scalar_v3.csv"))

# get path reletave the current file
min_times_Metal = DataFrame()
if isfile(Metal_file) 
  Metal_scalar = DataFrame(CSV.File(Metal_file))
  if !isempty(Metal_scalar)
    min_times_Metal = combine(groupby(Metal_scalar, [:N, :type, :op]), "times" => minimum => :min_time)
    min_times_Metal[!, :impl] .= "Metal.jl"
    min_times_Metal[!, :name] .= "Metal.jl "
  end
else
  @warn "Metal_scalar.csv not found"
end

min_times_KA_V1 = DataFrame()


min_times_KA_V3 = DataFrame()
if isfile(KA_V3_file) 
  KA_V3_scalar = DataFrame(CSV.File(KA_V3_file))
  if !isempty(KA_V3_scalar)
    min_times_KA_V3 = combine(groupby(KA_V3_scalar, [:N, :type, :op]), "times" => minimum => :min_time)
    min_times_KA_V3[!, :name] .= "Vendor neutral 2 "
    min_times_KA_V3[!, :impl] .= "Vendor neutral 2"
  end
else
  @warn "KA_scalar_v3.csv not found"
end

merged_df = vcat(min_times_Metal, min_times_KA_V3)
if !isempty(merged_df)
  merged_df[!, :min_time] = merged_df[!, :min_time] ./ 1000
end

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
          plot!(group.N, group.min_time, label="$name", marker=:circle, color= (impl == "Metal.jl" ? :red : impl == "Vendor neutral 1" ? :blue : :green))
          end

          png(joinpath(path, joinpath("scalar/"*type*"_"*op*".png")))
      end
  end
end