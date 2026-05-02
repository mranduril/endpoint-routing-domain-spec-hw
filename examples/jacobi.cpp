#include "job.h"
#include "runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
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

std::string run_timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_r(&now_time, &local_time);

    std::ostringstream out;
    out << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    return out.str();
}
}

int main()
{
    constexpr std::size_t nx = 16;
    constexpr std::size_t ny = 12;
    constexpr std::size_t halo_width = 1;

    mkdir("outputs", 0775);
    mkdir("outputs/sim_traces", 0775);
    const std::string timestamp = run_timestamp();
    const char* trace_path = "outputs/sim_traces/routing_jacobi_sim_jobs.jsonl";
    const std::string log_path = "outputs/routing_jacobi_run_log_" +
        timestamp + ".jsonl";
    std::ofstream(trace_path, std::ios::trunc).close();
    std::ofstream(log_path, std::ios::trunc).close();
    setenv("ROUTING_SIM_TRACE", trace_path, 1);
    setenv("ROUTING_RUN_LOG", log_path.c_str(), 1);

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

    std::fill(output->begin(), output->end(), 0.0f);

    metadata.job_id = 3;
    auto sim_interior = Routing::make_jacobi(
        Routing::WorkloadType::JACOBI_INTERIOR,
        nx,
        ny,
        halo_width,
        input,
        output,
        metadata);
    Routing::Request sim_interior_request =
        Routing::submit(sim_interior, Routing::RoutingPolicy::ForceSim);
    sim_interior_request.wait();

    metadata.job_id = 4;
    auto sim_boundary = Routing::make_jacobi(
        Routing::WorkloadType::JACOBI_BOUNDARY,
        nx,
        ny,
        halo_width,
        input,
        output,
        metadata);
    Routing::Request sim_boundary_request =
        Routing::submit(sim_boundary, Routing::RoutingPolicy::ForceSim);
    sim_boundary_request.wait();

    metadata.job_id = 5;
    auto sim_halo_boundary = Routing::make_jacobi(
        Routing::WorkloadType::JACOBI_HALO_BOUNDARY,
        nx,
        ny,
        halo_width,
        input,
        output,
        metadata);
    Routing::Request sim_halo_boundary_request =
        Routing::submit(sim_halo_boundary, Routing::RoutingPolicy::ForceSim);
    sim_halo_boundary_request.wait();

    if (!verify_jacobi(*input, *output, nx, ny)) {
        std::cerr << "Jacobi SIM verification failed\n";
        return 1;
    }

    std::cout << "Jacobi CPU verification passed\n";
    std::cout << "Jacobi SIM verification passed\n";
    std::cout << "SIM trace emitted to " << trace_path << '\n';
    std::cout << "Run log emitted to " << log_path << '\n';
    return 0;
}
