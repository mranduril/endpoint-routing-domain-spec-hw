#include "kernels.h"

#include <mpi.h>

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
            // Boundary work means the band near the local tile edges. These
            // cells are the ones that matter for halo-aware scheduling.
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
                // Interior and boundary kernels are complementary; together
                // they update the full non-halo Jacobi domain exactly once.
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

    void jacobi_exchange_halos_cpu(
        std::size_t nx,
        std::size_t owned_rows,
        float* tile,
        int node_id,
        int neighbor_node_id)
    {
        if (tile == nullptr) {
            throw std::invalid_argument("jacobi_exchange_halos_cpu received a null tile");
        }

        if (nx == 0 || owned_rows == 0 || neighbor_node_id < 0) {
            return;
        }

        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            throw std::runtime_error("MPI must be initialized before Jacobi halo exchange");
        }

        constexpr int upper_to_lower_tag = 100;
        constexpr int lower_to_upper_tag = 101;

        if (node_id < neighbor_node_id) {
            MPI_Sendrecv(
                tile + owned_rows * nx,
                static_cast<int>(nx),
                MPI_FLOAT,
                neighbor_node_id,
                upper_to_lower_tag,
                tile + (owned_rows + 1) * nx,
                static_cast<int>(nx),
                MPI_FLOAT,
                neighbor_node_id,
                lower_to_upper_tag,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
            return;
        }

        MPI_Sendrecv(
            tile + nx,
            static_cast<int>(nx),
            MPI_FLOAT,
            neighbor_node_id,
            lower_to_upper_tag,
            tile,
            static_cast<int>(nx),
            MPI_FLOAT,
            neighbor_node_id,
            upper_to_lower_tag,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    }
}
