#include "CUB_ReductionBenchmark.cuh"

#include <cub/cub.cuh>
#include <random>
#include <cstdint>
#include <fstream>
#include <iostream>

using namespace std;
typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;

// product function object
struct Product
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a * b;
    }
};


static void writeCSVHeader(string fileName){
    std::ofstream outputFile{fileName};
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        //exit(EXIT_FAILURE);
    }
    outputFile << "N;sizetype;type;elapsed;operation;result" << std::endl;
    outputFile.close();
}

int main()
{ 
    string fileName = "output.csv";
    writeCSVHeader(fileName);

    default_random_engine generator;
    
    uniform_int_distribution<uint8_t> uint8distribution(0, 5);
    uniform_int_distribution<uint16_t> uint16distribution(0, 5);
    uniform_int_distribution<uint32_t> uint32distribution(0, 5);
    uniform_int_distribution<uint64_t> uint64distribution(0, 5);
    uniform_int_distribution<uint128_t> uint128distribution(0, 5);
    

    uniform_int_distribution<int8_t> int8distribution(0, 5);
    uniform_int_distribution<int16_t> int16distribution(0, 5);
    uniform_int_distribution<int32_t> int32distribution(0, 5);
    uniform_int_distribution<int64_t> int64distribution(0, 5);
    uniform_int_distribution<int128_t> int128distribution(0, 5);

    uniform_real_distribution<float> floatdistribution(0.0, 1.0);
    uniform_real_distribution<double> doubledistribution(0.0, 1.0);


    // ########################################
    // Sum
    // // ########################################
    // CUB_ReductionBenchmark< uint8_t,
    //                                 decltype(cub::Sum()),
    //                                 uniform_int_distribution<uint8_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Sum(), uint8distribution, generator, "uint8_t", "sum",0).run();
    // CUB_ReductionBenchmark< uint16_t,
    //                                 decltype(cub::Sum()),
    //                                 uniform_int_distribution<uint16_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Sum(), uint16distribution, generator, "uint16_t", "sum",0).run();
    // CUB_ReductionBenchmark< uint32_t,
    //                                 decltype(cub::Sum()),
    //                                 uniform_int_distribution<uint32_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Sum(), uint32distribution, generator, "uint32_t", "sum",0).run();
    CUB_ReductionBenchmark< uint64_t,
                                    decltype(cub::Sum()),
                                    uniform_int_distribution<uint64_t>,
                                    default_random_engine>
        (fileName, cub::Sum(), uint64distribution, generator, "uint64_t", "sum",0).run();
    // CUB_ReductionBenchmark< uint128_t,
    //                                 decltype(cub::Sum()),
    //                                 uniform_int_distribution<uint128_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Sum(), uint128distribution, generator, "uint128_t", "sum",0).run();
    
    // CUB_ReductionBenchmark< int8_t,
    //                                 decltype(cub::Sum()),
    //                                 uniform_int_distribution<int8_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Sum(), int8distribution, generator, "int8_t", "sum",0).run();
    // CUB_ReductionBenchmark< int16_t,
    //                                 decltype(cub::Sum()),
    //                                 uniform_int_distribution<int16_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Sum(), int16distribution, generator, "int16_t", "sum",0).run();
    // CUB_ReductionBenchmark< int32_t,
    //                                 decltype(cub::Sum()),
    //                                 uniform_int_distribution<int32_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Sum(), int32distribution, generator, "int32_t", "sum",0).run();
    // CUB_ReductionBenchmark< int64_t,
    //                                 decltype(cub::Sum()),
    //                                 uniform_int_distribution<int64_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Sum(), int64distribution, generator, "int64_t", "sum",0).run();
    // CUB_ReductionBenchmark< int128_t,
    //                                 decltype(cub::Sum()),
    //                                 uniform_int_distribution<int128_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Sum(), int128distribution, generator, "int128_t", "sum",0).run();
    
    // CUB_ReductionBenchmark< float,
    //                                 decltype(cub::Sum()),
    //                                 uniform_real_distribution<float>,
    //                                 default_random_engine>
    //     (fileName, cub::Sum(), floatdistribution, generator, "float", "sum",0).run();
    // CUB_ReductionBenchmark< double,
    //                                 decltype(cub::Sum()),
    //                                 uniform_real_distribution<double>,
    //                                 default_random_engine>
    //     (fileName, cub::Sum(), doubledistribution, generator, "double", "sum",0).run();


    // // ########################################
    // // Minimum
    // // ########################################  
    // CUB_ReductionBenchmark< uint8_t,
    //                                 decltype(cub::Min()),
    //                                 uniform_int_distribution<uint8_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), uint8distribution, generator, "uint8_t", "min",std::numeric_limits<uint8_t>::max()).run();
    // CUB_ReductionBenchmark< uint16_t,
    //                                 decltype(cub::Min()),
    //                                 uniform_int_distribution<uint16_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), uint16distribution, generator, "uint16_t", "min",std::numeric_limits<uint16_t>::max()).run();
    // CUB_ReductionBenchmark< uint32_t,
    //                                 decltype(cub::Min()),
    //                                 uniform_int_distribution<uint32_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), uint32distribution, generator, "uint32_t", "min",std::numeric_limits<uint32_t>::max()).run();
    // CUB_ReductionBenchmark< uint64_t,
    //                                 decltype(cub::Min()),
    //                                 uniform_int_distribution<uint64_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), uint64distribution, generator, "uint64_t", "min",std::numeric_limits<uint64_t>::max()).run();
    // CUB_ReductionBenchmark< uint128_t,
    //                                 decltype(cub::Min()),
    //                                 uniform_int_distribution<uint128_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), uint128distribution, generator, "uint128_t", "min",std::numeric_limits<uint128_t>::max()).run();
    
    // CUB_ReductionBenchmark< int8_t,
    //                                 decltype(cub::Min()),
    //                                 uniform_int_distribution<int8_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), int8distribution, generator, "int8_t", "min",std::numeric_limits<int8_t>::max()).run();
    // CUB_ReductionBenchmark< int16_t,
    //                                 decltype(cub::Min()),
    //                                 uniform_int_distribution<int16_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), int16distribution, generator, "int16_t", "min",std::numeric_limits<int16_t>::max()).run();
    // CUB_ReductionBenchmark< int32_t,
    //                                 decltype(cub::Min()),
    //                                 uniform_int_distribution<int32_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), int32distribution, generator, "int32_t", "min",std::numeric_limits<int32_t>::max()).run();
    // CUB_ReductionBenchmark< int64_t,
    //                                 decltype(cub::Min()),
    //                                 uniform_int_distribution<int64_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), int64distribution, generator, "int64_t", "min",std::numeric_limits<int64_t>::max()).run();
    // CUB_ReductionBenchmark< int128_t,
    //                                 decltype(cub::Min()),
    //                                 uniform_int_distribution<int128_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), int128distribution, generator, "int128_t", "min",std::numeric_limits<int128_t>::max()).run();
    
    // CUB_ReductionBenchmark< float,
    //                                 decltype(cub::Min()),
    //                                 uniform_real_distribution<float>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), floatdistribution, generator, "float", "min",std::numeric_limits<float>::max()).run();
    // CUB_ReductionBenchmark< double,
    //                                 decltype(cub::Min()),
    //                                 uniform_real_distribution<double>,
    //                                 default_random_engine>
    //     (fileName, cub::Min(), doubledistribution, generator, "double", "min",std::numeric_limits<double>::max()).run();
    
    // // ########################################
    // // Maximum
    // // ########################################  
    // CUB_ReductionBenchmark< uint8_t,
    //                                 decltype(cub::Max()),
    //                                 uniform_int_distribution<uint8_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), uint8distribution, generator, "uint8_t", "max",std::numeric_limits<uint8_t>::min()).run();
    // CUB_ReductionBenchmark< uint16_t,
    //                                 decltype(cub::Max()),
    //                                 uniform_int_distribution<uint16_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), uint16distribution, generator, "uint16_t", "max",std::numeric_limits<uint16_t>::min()).run();
    // CUB_ReductionBenchmark< uint32_t,
    //                                 decltype(cub::Max()),
    //                                 uniform_int_distribution<uint32_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), uint32distribution, generator, "uint32_t", "max",std::numeric_limits<uint32_t>::min()).run();
    // CUB_ReductionBenchmark< uint64_t,
    //                                 decltype(cub::Max()),
    //                                 uniform_int_distribution<uint64_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), uint64distribution, generator, "uint64_t", "max",std::numeric_limits<uint64_t>::min()).run();
    // CUB_ReductionBenchmark< uint128_t,
    //                                 decltype(cub::Max()),
    //                                 uniform_int_distribution<uint128_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), uint128distribution, generator, "uint128_t", "max",std::numeric_limits<uint128_t>::min()).run();
    
    // CUB_ReductionBenchmark< int8_t,
    //                                 decltype(cub::Max()),
    //                                 uniform_int_distribution<int8_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), int8distribution, generator, "int8_t", "max",std::numeric_limits<int8_t>::min()).run();
    // CUB_ReductionBenchmark< int16_t,
    //                                 decltype(cub::Max()),
    //                                 uniform_int_distribution<int16_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), int16distribution, generator, "int16_t", "max",std::numeric_limits<int16_t>::min()).run();
    // CUB_ReductionBenchmark< int32_t,
    //                                 decltype(cub::Max()),
    //                                 uniform_int_distribution<int32_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), int32distribution, generator, "int32_t", "max",std::numeric_limits<int32_t>::min()).run();
    // CUB_ReductionBenchmark< int64_t,
    //                                 decltype(cub::Max()),
    //                                 uniform_int_distribution<int64_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), int64distribution, generator, "int64_t", "max",std::numeric_limits<int64_t>::min()).run();
    // CUB_ReductionBenchmark< int128_t,
    //                                 decltype(cub::Max()),
    //                                 uniform_int_distribution<int128_t>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), int128distribution, generator, "int128_t", "max",std::numeric_limits<int128_t>::min()).run();
    
    // CUB_ReductionBenchmark< float,
    //                                 decltype(cub::Max()),
    //                                 uniform_real_distribution<float>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), floatdistribution, generator, "float", "max",std::numeric_limits<float>::min()).run();
    // CUB_ReductionBenchmark< double,
    //                                 decltype(cub::Max()),
    //                                 uniform_real_distribution<double>,
    //                                 default_random_engine>
    //     (fileName, cub::Max(), doubledistribution, generator, "double", "max",std::numeric_limits<double>::min()).run();

    // // ########################################
    // // Product
    // // ######################################## 
    // Product product;
    // CUB_ReductionBenchmark< uint8_t,
    //                                 Product,
    //                                 uniform_int_distribution<uint8_t>,
    //                                 default_random_engine>
    //     (fileName, product, uint8distribution, generator, "uint8_t", "product",1).run();
    // CUB_ReductionBenchmark< uint16_t,
    //                                 Product,
    //                                 uniform_int_distribution<uint16_t>,
    //                                 default_random_engine>
    //     (fileName, product, uint16distribution, generator, "uint16_t", "product",1).run();
    // CUB_ReductionBenchmark< uint32_t,
    //                                 Product,
    //                                 uniform_int_distribution<uint32_t>,
    //                                 default_random_engine>
    //     (fileName, product, uint32distribution, generator, "uint32_t", "product",1).run();
    // CUB_ReductionBenchmark< uint64_t,
    //                                 Product,
    //                                 uniform_int_distribution<uint64_t>,
    //                                 default_random_engine>
    //     (fileName, product, uint64distribution, generator, "uint64_t", "product",1).run();
    // CUB_ReductionBenchmark< uint128_t,
    //                                 Product,
    //                                 uniform_int_distribution<uint128_t>,
    //                                 default_random_engine>
    //     (fileName, product, uint128distribution, generator, "uint128_t", "product",1).run();

    // CUB_ReductionBenchmark< int8_t,
    //                                 Product,
    //                                 uniform_int_distribution<int8_t>,
    //                                 default_random_engine>
    //     (fileName, product, int8distribution, generator, "int8_t", "product",1).run();
    // CUB_ReductionBenchmark< int16_t,
    //                                 Product,
    //                                 uniform_int_distribution<int16_t>,
    //                                 default_random_engine>
    //     (fileName, product, int16distribution, generator, "int16_t", "product",1).run();
    // CUB_ReductionBenchmark< int32_t,
    //                                 Product,
    //                                 uniform_int_distribution<int32_t>,
    //                                 default_random_engine>
    //     (fileName, product, int32distribution, generator, "int32_t", "product",1).run();
    // CUB_ReductionBenchmark< int64_t,
    //                                 Product,
    //                                 uniform_int_distribution<int64_t>,
    //                                 default_random_engine>
    //     (fileName, product, int64distribution, generator, "int64_t", "product",1).run();
    // CUB_ReductionBenchmark< int128_t,
    //                                 Product,
    //                                 uniform_int_distribution<int128_t>,
    //                                 default_random_engine>
    //     (fileName, product, int128distribution, generator, "int128_t", "product",1).run();

    // CUB_ReductionBenchmark< float,
    //                                 Product,
    //                                 uniform_real_distribution<float>,
    //                                 default_random_engine>
    //     (fileName, product, floatdistribution, generator, "float", "product",1).run();
    // CUB_ReductionBenchmark< double,
    //                                 Product,
    //                                 uniform_real_distribution<double>,
    //                                 default_random_engine>
    //     (fileName, product, doubledistribution, generator, "double", "product",1).run();   
    
    return 0;
}