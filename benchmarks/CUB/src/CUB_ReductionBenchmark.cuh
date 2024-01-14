#include "CUB_ReductionSample.h"

#include <string>
#include <random>
#include <cstdint>

using namespace std;

template <typename T, typename ReductionOpT, typename DistributionT, typename GeneratorT>
class CUB_ReductionBenchmark 
{ 
  private:
    string m_FileName;
    ReductionOpT m_ReductionOp;
    DistributionT m_Distribution;
    GeneratorT m_Generator;
    string m_TypeName;
    string m_OperationName;
    T m_NeutralValue;

  public:
    CUB_ReductionBenchmark(string filename, ReductionOpT reduction_op, DistributionT distribution, GeneratorT generator, string type_name, string operation_name, T neutral_value) : 
                                                        m_FileName(filename), 
                                                        m_ReductionOp(reduction_op), 
                                                        m_Distribution(distribution), 
                                                        m_Generator(generator), 
                                                        m_TypeName(type_name), 
                                                        m_OperationName(operation_name),
                                                        m_NeutralValue(neutral_value) {}

  void run()
  {  
    for (int n = 256; n < 5000000; n=n*2)
    {
        std::ofstream outputFile{m_FileName, std::ios::app};

        if (!outputFile.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            //exit(EXIT_FAILURE);
        }

        for(int i = 0; i < 50; i++)
        {       
          auto s = create_sample(n);
            outputFile << s << std::endl;
        }
        outputFile.close();
    }
  };


  private:
  CUB_ReductionSample<T> create_sample(int n)
  {
    auto size = sizeof(T);
    T result;
    float time = 0;

    T* values = new T[n];

    T*  d_in = nullptr;
    T*  d_out = nullptr;
    void*   d_temp_storage = NULL;
    size_t  temp_storage_bytes = 0;
    
    for (int i = 0; i < n; i++) values[i] = m_Distribution(m_Generator);

    cudaMalloc((void**)&d_in, n * sizeof(T));
    cudaMalloc((void**)&d_out, sizeof(T));
    cudaMemcpy(d_in, values, n * sizeof(T), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, n, m_ReductionOp, m_NeutralValue);
    cudaMalloc((T**)&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, n, m_ReductionOp, m_NeutralValue);

    cudaMemcpy(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
   
    cudaFree(d_temp_storage);
    cudaFree(d_in);
    cudaFree(d_out);
    
    delete[] values;

    time *= 1000;

    return CUB_ReductionSample<T>(n,size, m_TypeName, time, m_OperationName, result);
  };
};