// This should be wrappers that call the endpoints
#ifndef KERNELS_H
#define KERNELS_H

#include <cstddef>

namespace Routing
{
    void saxpy_cpu_only(float a, std::size_t n, const float* x, float* y);
    void saxpy_gpu_only(float a, std::size_t n, const float* x, float* y);
    // For demonstration purpose, SimPy may not be an endpoint that runs on its own.
    void saxpy_simpy(float a, std::size_t n, const float* x, float* y);
}

#endif // kernels.h