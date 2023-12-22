using CSV
using DataFrames
using Plots
using StatsPlots
using Plots.PlotMeasures


path = dirname(@__FILE__)
const CUDA_file = path*"/../benchmarks/CUDA/CUDA_scalar.csv"
const KA_V1_file = path*"/../benchmarks/CUDA/KA_scalar_v1.csv"
const CUB_file = path*"/../benchmarks/CUDA/CUB.csv"

# get path reletave the current file


CUDA_scalar = DataFrame(CSV.File(CUDA_file))
#KA_V1_scalar = DataFrame(CSV.File(KA_V1_file))
KA_V1_scalar = DataFrame(CSV.File(KA_V1_file))
CUB = DataFrame(CSV.File(CUB_file))

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

min_times_CUDA = combine(groupby(CUDA_scalar, [:N, :type, :op]), "times" => minimum => :min_time)
min_times_KA_V1 = combine(groupby(KA_V1_scalar, [:N, :type, :op]), "times" => minimum => :min_time)
min_times_CUB = combine(groupby(CUB, [:N, :type, :op]), "times" => minimum => :min_time)


min_times_CUDA[!, :name] .= "CUDA.jl "
min_times_KA_V1[!, :name] .= "Vendor neutral "
min_times_CUB[!, :name] .= "CUB "


min_times_CUDA[!, :impl] .= "CUDA.jl"
min_times_KA_V1[!, :impl] .= "Vendor neutral"
min_times_CUB[!, :impl] .= "CUB"

merged_df = vcat(min_times_CUDA, min_times_KA_V1, min_times_CUB)
merged_df[!, :min_time] = merged_df[!, :min_time] ./ 1000

merged_df[!, :op] = replace.(merged_df[!, :op], "+" => "sum")

merged_df[!, :op] = replace.(merged_df[!, :op], "*" => "prod")

#get unique types
types = unique(merged_df[!, :type])
#get unique operations
ops = unique(merged_df[!, :op])

#iteratore over each combination of type and operation
for type in types
    for op in ops
        #filter the dataframe for the current type and operation
        plot(xaxis=:log2, title="Minimum execution time scalar "*op*" reduction "*type, xlabel="N", ylabel="Time (μs)", legend=:topleft, size=(800, 600), link=:both, left_margin = [20mm 0mm], bottom_margin = [20mm 0mm])

        filtered_df = filter(row -> row[:type] == type && row[:op] == op, merged_df)
        grouped_df = groupby(filtered_df, [:name, :impl, :type, :op])



        for group in grouped_df

         name, impl = group[1, :name], group[1, :impl]
         display(name)
         plot!(group.N, group.min_time, label="$name", marker=:auto, color= (impl == "CUDA.jl" ? :red : impl == "CUB" ? :blue : :green))
        end
        savefig(path*"/img/CUDA/CUDA_scalar_"*type*"_"*op*".png")
    end
end

# Create a plot

# Iterate through each group and add a line with a unique color and marker

