#include "job.h"
#include "runtime.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace
{
float expected_cell(const std::vector<float>& input, std::size_t nx, std::size_t row, std::size_t col)
{
    return 0.25f * (
        input[(row - 1) * nx + col] +
        input[(row + 1) * nx + col] +
        input[row * nx + (col - 1)] +
        input[row * nx + (col + 1)]);
}

bool verify_jacobi(
    const std::vector<float>& input,
    const std::vector<float>& output,
    std::size_t nx,
    std::size_t ny)
{
    constexpr float epsilon = 1.0e-5f;
    for (std::size_t row = 1; row + 1 < ny; ++row) {
        for (std::size_t col = 1; col + 1 < nx; ++col) {
            const float expected = expected_cell(input, nx, row, col);
            const float actual = output[row * nx + col];
            if (std::fabs(expected - actual) > epsilon) {
                std::cerr << "Mismatch at (" << row << ", " << col
                          << "): expected " << expected
                          << ", got " << actual << '\n';
                return false;
            }
        }
    }
    return true;
}
}

int main()
{
    constexpr std::size_t nx = 16;
    constexpr std::size_t ny = 12;
    constexpr std::size_t halo_width = 1;

    auto input = std::make_shared<std::vector<float>>(nx * ny);
    auto output = std::make_shared<std::vector<float>>(nx * ny, 0.0f);
    for (std::size_t row = 0; row < ny; ++row) {
        for (std::size_t col = 0; col < nx; ++col) {
            (*input)[row * nx + col] = static_cast<float>(row + col);
        }
    }

    Routing::JobMetadata metadata;
    metadata.job_id = 1;
    metadata.node_id = 0;
    metadata.iteration = 0;

    auto interior = Routing::make_jacobi(
        Routing::WorkloadType::JACOBI_INTERIOR,
        nx,
        ny,
        halo_width,
        input,
        output,
        metadata);
    Routing::Request interior_request =
        Routing::submit(interior, Routing::RoutingPolicy::ForceCpu);
    interior_request.wait();

    metadata.job_id = 2;
    auto boundary = Routing::make_jacobi(
        Routing::WorkloadType::JACOBI_BOUNDARY,
        nx,
        ny,
        halo_width,
        input,
        output,
        metadata);
    Routing::Request boundary_request =
        Routing::submit(boundary, Routing::RoutingPolicy::ForceCpu);
    boundary_request.wait();

    if (!verify_jacobi(*input, *output, nx, ny)) {
        return 1;
    }

    const char* trace_path = "/tmp/routing_jacobi_sim_jobs.jsonl";
    std::ofstream(trace_path, std::ios::trunc).close();
    setenv("ROUTING_SIM_TRACE", trace_path, 1);

    metadata.job_id = 3;
    metadata.arrival_us = 100.0;
    auto sim_boundary = Routing::make_jacobi(
        Routing::WorkloadType::JACOBI_HALO_BOUNDARY,
        nx,
        ny,
        halo_width,
        input,
        output,
        metadata);
    Routing::Request sim_request =
        Routing::submit(sim_boundary, Routing::RoutingPolicy::ForceSim);
    sim_request.wait();

    std::cout << "Jacobi CPU verification passed\n";
    std::cout << "SIM trace emitted to " << trace_path << '\n';
    return 0;
}
