#include "kernels.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>


void throw_cuda_error(cudaError_t status, const char* context)
{
    if (status == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string(context) + ": " + cudaGetErrorString(status));
}

__global__ void saxpy_kernel(std::size_t n, float a, const float* x, float* y)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}


namespace Routing
{

void saxpy_gpu_only(float a, std::size_t n, const float* x, float* y)
{
    if (x == nullptr || y == nullptr) {
        throw std::invalid_argument("saxpy_gpu_only received a null buffer");
    }

    const std::size_t bytes = n * sizeof(float);
    float* d_x = nullptr;
    float* d_y = nullptr;

    auto cleanup = [&]() {
        if (d_x != nullptr) {
            cudaFree(d_x);
        }
        if (d_y != nullptr) {
            cudaFree(d_y);
        }
    };

    try {
        throw_cuda_error(cudaMalloc(&d_x, bytes), "cudaMalloc(d_x)");
        throw_cuda_error(cudaMalloc(&d_y, bytes), "cudaMalloc(d_y)");

        throw_cuda_error(
            cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D x");
        throw_cuda_error(
            cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D y");

        constexpr int block_size = 256;
        const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
        saxpy_kernel<<<grid_size, block_size>>>(n, a, d_x, d_y);

        throw_cuda_error(cudaGetLastError(), "saxpy_kernel launch");
        throw_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        throw_cuda_error(
            cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H y");
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
}

} // namespace Routing
