// This should be wrappers that call the endpoints
#ifndef KERNELS_H
#define KERNELS_H

#include "workload_type.h"

#include <cstddef>

namespace Routing
{
    void saxpy_cpu_only(float a, std::size_t n, const float* x, float* y);
    void saxpy_gpu_only(float a, std::size_t n, const float* x, float* y);
    void saxpy_simpy(
        float a,
        std::size_t n,
        const float* x,
        float* y,
        const JobMetadata& metadata = {});

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
    void jacobi_interior_simpy(
        std::size_t nx,
        std::size_t ny,
        std::size_t halo_width,
        const float* input,
        float* output,
        const JobMetadata& metadata = {});
    void jacobi_boundary_simpy(
        std::size_t nx,
        std::size_t ny,
        std::size_t halo_width,
        const float* input,
        float* output,
        const JobMetadata& metadata = {});
    void jacobi_halo_boundary_simpy(
        std::size_t nx,
        std::size_t ny,
        std::size_t halo_width,
        const float* input,
        float* output,
        const JobMetadata& metadata = {});
}

#endif // kernels.h
