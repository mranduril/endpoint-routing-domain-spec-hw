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

    void jacobi_interior_cpu(
        std::size_t nx,
        std::size_t ny,
        std::size_t halo_width,
        const float* input,
        float* output);
    void jacobi_boundary_cpu(
        std::size_t nx,
        std::size_t ny,
        std::size_t halo_width,
        const float* input,
        float* output);
    void jacobi_interior_gpu(
        std::size_t nx,
        std::size_t ny,
        std::size_t halo_width,
        const float* input,
        float* output);
}

#endif // kernels.h
