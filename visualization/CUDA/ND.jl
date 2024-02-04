using CSV
using DataFrames
using Plots
using StatsPlots
using Plots.PlotMeasures

path = dirname(@__FILE__)
const CUDA_file = joinpath(path, joinpath("../../benchmarks/CUDA/CUDA_ND_elements_per_group.csv"))
const KA_V1_file = joinpath(path, joinpath("../../benchmarks/CUDA/KA_ND_elements_per_group_v1.csv"))
const KA_V3_file = joinpath(path, joinpath("../../benchmarks/CUDA/KA_ND_elements_per_group_v3.csv"))

min_times_CUDA = DataFrame()
if isfile(CUDA_file)
    CUDA_scalar = DataFrame(CSV.File(CUDA_file))
    if !isempty(CUDA_scalar)
      min_times_CUDA = combine(groupby(CUDA_scalar, [:N, :type, :op]), "times" => minimum => :min_time)
      min_times_CUDA[!, :name] .= "CUDA.jl "
      min_times_CUDA[!, :impl] .= "CUDA.jl"
    end
else
    @warn "CUDA_ND_elements_per_group.csv not found"
end

min_times_KA_V1 = DataFrame()
# if isfile(KA_V1_file)
#     KA_V1_scalar = DataFrame(CSV.File(KA_V1_file))
#     if !isempty(KA_V1_scalar)
#       min_times_KA_V1 = combine(groupby(KA_V1_scalar, [:N, :type,:op]), "times" => minimum => :min_time)
#       min_times_KA_V1[!, :name] .= "Vendor neutral 1 "
#       min_times_KA_V1[!, :impl] .= "Vendor neutral 1"
#     end
# else
#     @warn "KA_ND_elements_per_group_v1.csv not found"
# end

min_times_KA_V3 = DataFrame()
if isfile(KA_V3_file)
    KA_V3_scalar = DataFrame(CSV.File(KA_V3_file))
    if !isempty(KA_V3_scalar)
      min_times_KA_V3 = combine(groupby(KA_V3_scalar, [:N, :type, :op]), "times" => minimum => :min_time)
      min_times_KA_V3[!, :name] .= "Vendor neutral "
      min_times_KA_V3[!, :impl] .= "Vendor neutral"
    end
else
    @warn "KA_ND_elements_per_group_v3.csv not found"
end


merged_df = vcat(min_times_CUDA, min_times_KA_V3)
merged_df = vcat(merged_df, min_times_KA_V1)

if !isempty(merged_df)
  if !isdir(joinpath(path, "ND"))
    mkdir(joinpath(path, "ND"))
  end

  merged_df[!, :op] = replace.(merged_df[!, :op], "+" => "sum")
  merged_df[!, :op] = replace.(merged_df[!, :op], "*" => "prod")

  types = unique(merged_df[!, :type])
  ops = unique(merged_df[!, :op])

  merged_df[!, :min_time] = merged_df[!, :min_time] ./ 1000

  for type in types
    for op in ops
      plot(xaxis=:log2, title="Minimum execution time ND "*op*" reduction "*type*"( N,32) to (1,32)", xlabel="N", ylabel="Time (μs)", legend=:topleft, size=(800, 600), link=:both, left_margin = [10mm 0mm], bottom_margin = [10mm 0mm], right_margin = [10mm 0mm])

      filtered_df = filter(row -> row[:type] == type && row[:op] == op, merged_df)
      grouped_df = groupby(filtered_df, [:name, :impl, :type, :op])

      # Iterate through each group and add a line with a unique color and marker
      for group in grouped_df
          name, impl = group[1, :name], group[1, :impl]
          plot!(group.N, group.min_time, label="$name", marker=:circle, color= (impl == "CUDA.jl" ? :red : impl == "Vendor neutral 1" ? :blue : :green))
      end
      png(joinpath(path, joinpath("ND/"*type*"_"*op*".png")))
    end
  end
end