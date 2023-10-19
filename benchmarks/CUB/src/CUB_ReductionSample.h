#include <string>
#include <fstream>
#include <iostream>
#include <cuda/functional>
#include <cub/cub.cuh>
#include <cstdint>

using namespace std;

typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;

string convert(const float& value) { return to_string(value); }
string convert(const double& value) { return to_string(value); }

string convert(const uint8_t& value) { return to_string(unsigned(value)); }
string convert(const uint16_t& value) { return to_string(unsigned(value)); }
string convert(const uint32_t& value) { return to_string(unsigned(value)); }
string convert(const uint64_t& value) { return to_string(unsigned(value)); }
string convert(const uint128_t& value) { return to_string(unsigned(value)); }


string convert(const int8_t& value) { return to_string(int(value)); }
string convert(const int16_t& value) { return to_string(int(value)); }
string convert(const int32_t& value) { return to_string(int(value)); }
string convert(const int64_t& value) { return to_string(int(value)); }
string convert(const int128_t& value) { return to_string(int(value)); }

template <class T>
class CUB_ReductionSample 
{
  private:
    uint32_t m_NumItems;          ///< Number of items to process
    uint8_t m_TypeSize;           ///< Size of the type
    string mTypeName;             ///< Type name
    float m_ElapsedMicroSeconds;  ///< Elapsed time in microseconds
    string m_Operator;            ///< Operator name
    T m_value;                      ///< value of reduction

  public:
  CUB_ReductionSample(uint32_t numItems, uint8_t typeSize, string typeName, float elapsedMicroSeconds, string op, T value) :
                                                    m_NumItems(numItems),
                                                    m_TypeSize(typeSize),
                                                    mTypeName(typeName),
                                                    m_ElapsedMicroSeconds(elapsedMicroSeconds),
                                                    m_Operator(op),
                                                    m_value(value) {}

  uint32_t  getNumItems() const { return m_NumItems; }
  uint8_t   getTypeSize() const { return m_TypeSize; }
  string    getTypeName() const { return mTypeName; }
  float     getElapsedMicroSeconds() const { return m_ElapsedMicroSeconds; }
  string    getOperator() const { return m_Operator; }
  T         getValue() const { return m_value; }
};

template <typename T>
ostream& operator<<(ostream& os, const CUB_ReductionSample<T>& sample){
    os << sample.getNumItems() << ";"
         << unsigned(sample.getTypeSize()) << ";"
         << sample.getTypeName() << ";"
         << sample.getElapsedMicroSeconds() << ";"
         << sample.getOperator() << ";"
         << convert(sample.getValue());
    return os;
}

