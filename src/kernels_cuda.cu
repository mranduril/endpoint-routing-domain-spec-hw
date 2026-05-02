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

__global__ void jacobi_interior_kernel(
    std::size_t nx,
    std::size_t ny,
    std::size_t halo_width,
    const float* input,
    float* output)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const std::size_t total = nx * ny;
    if (idx >= total) {
        return;
    }

    const std::size_t row = idx / nx;
    const std::size_t col = idx % nx;
    // GPU support currently covers interior work only. Boundary/halo work is
    // intentionally left to CPU or SIM because that is where communication
    // sensitivity is easiest to model.
    if (row < 2 * halo_width || col < 2 * halo_width ||
        row >= ny - 2 * halo_width || col >= nx - 2 * halo_width) {
        return;
    }

    output[idx] = 0.25f * (
        input[(row - 1) * nx + col] +
        input[(row + 1) * nx + col] +
        input[row * nx + (col - 1)] +
        input[row * nx + (col + 1)]);
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

void jacobi_interior_gpu(
    std::size_t nx,
    std::size_t ny,
    std::size_t halo_width,
    const float* input,
    float* output)
{
    // This simple wrapper copies full input/output grids each launch. The cost
    // model can skip these copies when metadata says data is already on GPU,
    // but this implementation still assumes host vectors for correctness.
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("jacobi_interior_gpu received a null buffer");
    }

    if (nx <= 2 * halo_width || ny <= 2 * halo_width) {
        return;
    }

    const std::size_t cells = nx * ny;
    const std::size_t bytes = cells * sizeof(float);
    float* d_input = nullptr;
    float* d_output = nullptr;

    auto cleanup = [&]() {
        if (d_input != nullptr) {
            cudaFree(d_input);
        }
        if (d_output != nullptr) {
            cudaFree(d_output);
        }
    };

    try {
        throw_cuda_error(cudaMalloc(&d_input, bytes), "cudaMalloc(d_input)");
        throw_cuda_error(cudaMalloc(&d_output, bytes), "cudaMalloc(d_output)");
        throw_cuda_error(
            cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D jacobi input");
        throw_cuda_error(
            cudaMemcpy(d_output, output, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D jacobi output");

        constexpr int block_size = 256;
        const int grid_size = static_cast<int>((cells + block_size - 1) / block_size);
        jacobi_interior_kernel<<<grid_size, block_size>>>(
            nx,
            ny,
            halo_width,
            d_input,
            d_output);

        throw_cuda_error(cudaGetLastError(), "jacobi_interior_kernel launch");
        throw_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        throw_cuda_error(
            cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H jacobi output");
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
}

} // namespace Routing
