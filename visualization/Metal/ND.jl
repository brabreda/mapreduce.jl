using CSV
using DataFrames
using Plots
using StatsPlots
using Plots.PlotMeasures

path = dirname(@__FILE__)
const Metal_file = joinpath(path, joinpath("../../benchmarks/Metal/Metal_ND_elements_per_group.csv"))
const KA_V1_file = joinpath(path, joinpath("../../benchmarks/Metal/KA_ND_elements_per_group_v1.csv"))
const KA_V3_file = joinpath(path, joinpath("../../benchmarks/Metal/KA_ND_elements_per_group_v3.csv"))

min_times_Metal = DataFrame()
if isfile(Metal_file)
    Metal_scalar = DataFrame(CSV.File(Metal_file))
    if !isempty(Metal_scalar)
      min_times_Metal = combine(groupby(Metal_scalar, [:N, :type, :op]), "times" => minimum => :min_time)
      min_times_Metal[!, :name] .= "Metal.jl "
      min_times_Metal[!, :impl] .= "Metal.jl"
    end
else
    @warn "Metal_ND_elements_per_group.csv not found"
end

min_times_KA_V1 = DataFrame()
if isfile(KA_V1_file)
    KA_V1_scalar = DataFrame(CSV.File(KA_V1_file))
    if !isempty(KA_V1_scalar)
      min_times_KA_V1 = combine(groupby(KA_V1_scalar, [:N, :type,:op]), "times" => minimum => :min_time)
      min_times_KA_V1[!, :name] .= "Vendor neutral 1 "
      min_times_KA_V1[!, :impl] .= "KernelAbstractions.jl 1"
    end
else
    @warn "KA_ND_elements_per_group_v1.csv not found"
end

min_times_KA_V3 = DataFrame()
if isfile(KA_V3_file)
    KA_V3_scalar = DataFrame(CSV.File(KA_V3_file))
    if !isempty(KA_V3_scalar)
      min_times_KA_V3 = combine(groupby(KA_V3_scalar, [:N, :type, :op]), "times" => minimum => :min_time)
      min_times_KA_V3[!, :name] .= "Vendor neutral 2 "
      min_times_KA_V3[!, :impl] .= "KernelAbstractions.jl 2"
    end
else
    @warn "KA_ND_elements_per_group_v3.csv not found"
end


merged_df = vcat(min_times_Metal, min_times_KA_V3)
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
          plot!(group.N, group.min_time, label="$name", marker=:circle, color= (impl == "Metal.jl" ? :red : impl == "Vendor neutral 1" ? :blue : :green))
      end
      png(joinpath(path, joinpath("ND/"*type*"_"*op*".png")))
    end
  end
end