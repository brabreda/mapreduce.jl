using CSV
using DataFrames
using Plots
using StatsPlots

path = dirname(@__FILE__)
const CUDA_file = path*"/../benchmarks/CUDA/CUDA_ND_elements_per_group.csv"
const KA_V2_file = path*"/../benchmarks/CUDA/KA_ND_elements_per_group_v2.csv"

CUDA_scalar = DataFrame(CSV.File(CUDA_file))
KA_V2_scalar = DataFrame(CSV.File(KA_V2_file))

min_times_CUDA = combine(groupby(CUDA_scalar, [:N, :type]), "times" => minimum => :min_time)
min_times_KA_V2 = combine(groupby(KA_V2_scalar, [:N, :type]), "times" => minimum => :min_time)

min_times_CUDA[!, :name] = "CUDA.jl " .* string.(min_times_CUDA[!, :type])
min_times_KA_V2[!, :name] = "Vendor neutral  " .* string.(min_times_KA_V2[!, :type])

min_times_CUDA[!, :impl] .= "CUDA.jl"
min_times_KA_V2[!, :impl] .= "KernelAbstractions.jl"

merged_df = vcat(min_times_CUDA, min_times_KA_V2)

merged_df[!, :min_time] = merged_df[!, :min_time] ./ 1000
merged_df = filter(row -> row[:type] in ["UInt32", "UInt64"], merged_df)

grouped_df = groupby(merged_df, [:name, :impl])

# Create a plot
plot(xaxis=:log2, title="Minimum execution time ND reduction for sum operator (N,32) to (1,32)", xlabel="N", ylabel="Time (μs)", legend=:topleft, size=(800, 600), link=:both)

# Iterate through each group and add a line with a unique color and marker
for group in grouped_df
    name, impl = group[1, :name], group[1, :impl]
    plot!(group.N, group.min_time, label="$name", marker=:auto, color= (impl == "CUDA.jl" ? :red : :blue))
end

savefig(path*"/img/CUDA_ND_elements_per_group.png")




#display(merged_df)