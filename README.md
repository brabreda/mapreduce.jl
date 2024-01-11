# mapreduce.jl

## setup
```julia
using Pkg
Pkg.add(url="https://github.com/brabreda/KernelAbstractions.jl.git")
```

## Benchmarks

you can run the benchmarks scripts with the following options:
- `KA`: runs benchmarks with vendor neutral implementation
- `CUDA`: runs benchmarks with CUDA.jl implementation
- `METAL`: runs benchmarks with METAL.jl implementation


- `UINT8`: runs benchmarks with UINT8 type
- `UINT16`: runs benchmarks with UINT16 type
- `UINT32`: runs benchmarks with UINT32 type
- `UINT64`: runs benchmarks with UINT64 type
- `UINT128`: runs benchmarks with UINT128 type

- `INT8`: runs benchmarks with INT8 type
- `INT16`: runs benchmarks with INT16 type
- `INT32`: runs benchmarks with INT32 type
- `INT64`: runs benchmarks with INT64 type
- `INT128`: runs benchmarks with INT128 type

- `FLOAT16`: runs benchmarks with FLOAT16 type
- `FLOAT32`: runs benchmarks with FLOAT32 type

- `SUM`: runs benchmarks with SUM operator
- `PROD`: runs benchmarks with PROD operator
- `MIN`: runs benchmarks with MIN operator
- `MAX`: runs benchmarks with MAX operator

Default (without arguments): `KA UINT8 UINT16 UINT32 UINT64 UINT128 INT8 INT16 INT32 INT64 INT128 FLOAT16 FLOAT32 SUM PROD MIN MAX`

### CUDA

```
julia benchmarks/CUDA/scalar.jl 
julia benchmarks/CUDA/ND.jl
```

### Metal
```
julia benchmarks/Metal/scalar.jl
julia benchmarks/Metal/ND.jl
```

## Visualisations

### CUDA
```
julia visualization/CUDA/scalar.jl
julia visualization/CUDA/ND.jl
```

### Metal
```
julia visualization/Metal/scalar.jl
julia visualization/Metal/ND.jl
```

## Visualisations

### CUDA
```
julia test/CUDATest.jl
```

### Metal
```
julia test/MetalTest.jl

```