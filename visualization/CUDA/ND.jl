using CSV
using DataFrames
using Plots
using StatsPlots
using Plots.PlotMeasures
using Statistics

path = dirname(@__FILE__)
const CUDA_file = joinpath(path, joinpath("../../benchmarks/CUDA/CUDA_ND_elements_per_group.csv"))
const KA_V1_file = joinpath(path, joinpath("../../benchmarks/CUDA/KA_ND_elements_per_group_v1.csv"))
const KA_V3_file = joinpath(path, joinpath("../../benchmarks/CUDA/KA_ND_elements_per_group_v3.csv"))

min_times_CUDA = DataFrame()
if isfile(CUDA_file)
    CUDA_scalar = DataFrame(CSV.File(CUDA_file))
    if !isempty(CUDA_scalar)
      min_times_CUDA = CUDA_scalar
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
      min_times_KA_V3 = KA_V3_scalar
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

  merged_df[!, :times] = merged_df[!, :times] ./ 1000

  merged_df = filter( row -> row[:N] < 2^28 && row[:gctimes] == 0.0, merged_df)
  #iteratore over each combination of type and operation
  for type in types
    for op in ops
      #filter the dataframe for the current type and operation
      plot(xaxis=:log2, title="Average execution time ND " * op * " reduction " * type* " ( N,32) to (1,32)", xlabel="N", ylabel="Time (μs)", legend=:topleft, size=(800, 600), link=:both, left_margin=[20mm 0mm], bottom_margin=[10mm 0mm], right_margin=[10mm 0mm])

      filtered_df = filter(row -> row[:type] == type && row[:op] == op, merged_df)
      grouped_df = groupby(filtered_df, [:name, :impl, :type, :op])


      for group in grouped_df
        average = combine(groupby(group, :N), "times" => mean => :times)
        ribbons = (combine(groupby(group, :N), "times" => std => :rib) .* 1.96) ./ sqrt.(combine(groupby(group, :N), nrow).nrow)
        display(combine(groupby(group, :N), "times" => std => :std))
        display(combine(groupby(group, :N), nrow).nrow)
        display(ribbons)

        name, impl = group[1, :name], group[1, :impl]
        plot!(unique(group.N), average.times, label="$name", marker=:circle, ribbon=ribbons.rib, color=(impl == "CUDA.jl" ? :red : impl == "CUB" ? :blue : impl == "Vendor neutral 2" ? :green : :orange))
      end

      png(joinpath(path, joinpath("ND/" * type * "_" * op * ".png")))
    end
  end
end