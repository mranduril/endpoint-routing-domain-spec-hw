// Endpoint wrappers. CPU/GPU functions perform real math. SIM functions emit a
// SimPy trace for performance modeling and then call CPU fallback for results.
#ifndef KERNELS_H
#define KERNELS_H

#include "workload_type.h"

#include <cstddef>

namespace Routing
{
    void saxpy_cpu_only(float a, std::size_t n, const float* x, float* y);
    void saxpy_gpu_only(float a, std::size_t n, const float* x, float* y);
    // SIM operators accept metadata because trace records need job/node IDs and
    // iteration information; CPU/GPU kernels intentionally stay math-only.
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
    void jacobi_exchange_halos_cpu(
        std::size_t nx,
        std::size_t owned_rows,
        float* tile,
        int node_id,
        int neighbor_node_id);
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
