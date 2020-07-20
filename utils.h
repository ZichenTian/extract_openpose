#include <string>

template <typename T>
struct Point {
    T x;
    T y;
};

inline unsigned int getNumberCudaBlocks(
    const unsigned int totalRequired, const unsigned int numberCudaThreads)
{
    return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
}

// template<typename T>
// inline void error(
//     const T& message, const int line = -1, const std::string& function = "", const std::string& file = "")
// {
//     error(tToString(message), line, function, file);
// }

// void cudaCheck(const int line, const std::string& function, const std::string& file)
//     {
//         try
//         {
//             #ifdef USE_CUDA
//                 const auto errorCode = cudaPeekAtLastError();
//                 if(errorCode != cudaSuccess)
//                     error("Cuda check failed (" + std::to_string(errorCode) + " vs. " + std::to_string(cudaSuccess) + "): "
//                           + cudaGetErrorString(errorCode), line, function, file);
//             #else
//                 UNUSED(line);
//                 UNUSED(function);
//                 UNUSED(file);
//                 error("OpenPose must be compiled with the `USE_CUDA` macro definition in order to use this"
//                       " functionality.", __LINE__, __FUNCTION__, __FILE__);
//             #endif
//         }
//         catch (const std::exception& e)
//         {
//             error(e.what(), __LINE__, __FUNCTION__, __FILE__);
//         }
//     }