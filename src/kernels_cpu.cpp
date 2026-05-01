#include "kernels.h"

#include <stdexcept>

namespace Routing
{
    namespace
    {
        float jacobi_cell(const float* input, std::size_t nx, std::size_t row, std::size_t col)
        {
            return 0.25f * (
                input[(row - 1) * nx + col] +
                input[(row + 1) * nx + col] +
                input[row * nx + (col - 1)] +
                input[row * nx + (col + 1)]);
        }

        bool is_boundary_work(
            std::size_t row,
            std::size_t col,
            std::size_t nx,
            std::size_t ny,
            std::size_t halo_width)
        {
            return row < 2 * halo_width ||
                col < 2 * halo_width ||
                row >= ny - 2 * halo_width ||
                col >= nx - 2 * halo_width;
        }
    }

    void saxpy_cpu_only(float a, std::size_t n, const float* x, float* y)
    {
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = a * x[i] + y[i];
        }
    }

    void saxpy_simpy(float a, std::size_t n, const float* x, float* y)
    {
        (void)a;
        (void)n;
        (void)x;
        (void)y;
        throw std::logic_error("SimPy SAXPY endpoint is not implemented yet");
    }

    void jacobi_interior_cpu(
        std::size_t nx,
        std::size_t ny,
        std::size_t halo_width,
        const float* input,
        float* output)
    {
        if (input == nullptr || output == nullptr) {
            throw std::invalid_argument("jacobi_interior_cpu received a null buffer");
        }

        if (nx <= 2 * halo_width || ny <= 2 * halo_width) {
            return;
        }

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (std::size_t row = halo_width; row < ny - halo_width; ++row) {
            for (std::size_t col = halo_width; col < nx - halo_width; ++col) {
                if (!is_boundary_work(row, col, nx, ny, halo_width)) {
                    output[row * nx + col] = jacobi_cell(input, nx, row, col);
                }
            }
        }
    }

    void jacobi_boundary_cpu(
        std::size_t nx,
        std::size_t ny,
        std::size_t halo_width,
        const float* input,
        float* output)
    {
        if (input == nullptr || output == nullptr) {
            throw std::invalid_argument("jacobi_boundary_cpu received a null buffer");
        }

        if (nx <= 2 * halo_width || ny <= 2 * halo_width) {
            return;
        }

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (std::size_t row = halo_width; row < ny - halo_width; ++row) {
            for (std::size_t col = halo_width; col < nx - halo_width; ++col) {
                if (is_boundary_work(row, col, nx, ny, halo_width)) {
                    output[row * nx + col] = jacobi_cell(input, nx, row, col);
                }
            }
        }
    }
}
