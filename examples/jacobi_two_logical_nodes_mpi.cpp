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
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace
{
struct LogicalNode {
    int node_id = 0;
    int neighbor_node_id = -1;
    std::size_t start_global_row = 0;
    std::size_t owned_rows = 0;
    std::size_t nx = 0;
    std::shared_ptr<std::vector<float>> input;
    std::shared_ptr<std::vector<float>> output;
    Routing::RouterConfig router_config;
};

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

bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }

    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on";
}

float initial_value(std::size_t row, std::size_t col)
{
    return static_cast<float>(row * 0.25 + col);
}

float expected_cell(const std::vector<float>& input, std::size_t nx, std::size_t row, std::size_t col)
{
    return 0.25f * (
        input[(row - 1) * nx + col] +
        input[(row + 1) * nx + col] +
        input[row * nx + (col - 1)] +
        input[row * nx + (col + 1)]);
}

std::vector<float> make_initial_grid(std::size_t nx, std::size_t ny)
{
    std::vector<float> grid(nx * ny);
    for (std::size_t row = 0; row < ny; ++row) {
        for (std::size_t col = 0; col < nx; ++col) {
            grid[row * nx + col] = initial_value(row, col);
        }
    }
    return grid;
}

std::vector<float> run_reference(
    std::vector<float> current,
    std::size_t nx,
    std::size_t ny,
    int iterations)
{
    std::vector<float> next = current;
    for (int iter = 0; iter < iterations; ++iter) {
        next = current;
        for (std::size_t row = 1; row + 1 < ny; ++row) {
            for (std::size_t col = 1; col + 1 < nx; ++col) {
                next[row * nx + col] = expected_cell(current, nx, row, col);
            }
        }
        current.swap(next);
    }
    return current;
}

void copy_row(
    const std::vector<float>& source,
    std::size_t source_row,
    std::vector<float>& destination,
    std::size_t destination_row,
    std::size_t nx)
{
    std::copy_n(
        source.data() + source_row * nx,
        nx,
        destination.data() + destination_row * nx);
}

void load_node_from_global(
    LogicalNode& node,
    const std::vector<float>& global,
    std::size_t global_ny)
{
    const std::size_t local_ny = node.owned_rows + 2;
    node.input = std::make_shared<std::vector<float>>(node.nx * local_ny);
    node.output = std::make_shared<std::vector<float>>(node.nx * local_ny);

    for (std::size_t local_row = 0; local_row < local_ny; ++local_row) {
        const std::size_t global_row = node.start_global_row + local_row - 1;
        if (global_row >= global_ny) {
            throw std::invalid_argument("logical node row partition exceeds global grid");
        }
        copy_row(global, global_row, *node.input, local_row, node.nx);
    }
    *node.output = *node.input;
}

Routing::RouterConfig make_logical_node_config(
    int local_node_id,
    bool gpu_available)
{
    Routing::RouterConfig config = Routing::make_router_config(
        local_node_id,
        Routing::CostModelPreset::SimPyAlignedStencil);
    config.endpoint_availability[local_node_id] =
        Routing::EndpointAvailability{true, gpu_available, true};
    return config;
}

Routing::Job make_jacobi_job(
    LogicalNode& node,
    Routing::WorkloadType type,
    std::size_t job_id,
    int iteration)
{
    Routing::JobMetadata metadata;
    metadata.job_id = job_id;
    metadata.node_id = node.node_id;
    metadata.iteration = iteration;
    metadata.neighbor_node_id = node.neighbor_node_id;

    return Routing::make_jacobi(
        type,
        node.nx,
        node.owned_rows + 2,
        1,
        node.input,
        node.output,
        metadata);
}

void route_iteration(
    const Routing::DistributedRuntime& runtime,
    LogicalNode& node,
    int iteration,
    std::size_t& next_job_id)
{
    runtime.exchange_jacobi_halos(
        node.nx,
        node.owned_rows,
        *node.input,
        node.neighbor_node_id);
    *node.output = *node.input;

    Routing::Request interior = Routing::submit(
        make_jacobi_job(
            node,
            Routing::WorkloadType::JACOBI_INTERIOR,
            next_job_id++,
            iteration),
        Routing::RoutingPolicy::Auto,
        node.router_config);
    interior.wait();

    Routing::Request boundary = Routing::submit(
        make_jacobi_job(
            node,
            Routing::WorkloadType::JACOBI_BOUNDARY,
            next_job_id++,
            iteration),
        Routing::RoutingPolicy::ForceCpu,
        node.router_config);
    boundary.wait();

    node.input.swap(node.output);
}

bool verify(
    const std::vector<float>& actual,
    const std::vector<float>& expected)
{
    constexpr float epsilon = 1.0e-5f;
    if (actual.size() != expected.size()) {
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > epsilon) {
            std::cerr << "Mismatch at flat index " << i
                      << ": expected " << expected[i]
                      << ", got " << actual[i] << '\n';
            return false;
        }
    }
    return true;
}
}

int main(int argc, char** argv)
{
    Routing::DistributedRuntime runtime(&argc, &argv);
    if (runtime.size() != 2) {
        if (runtime.rank() == 0) {
            std::cerr << "Run with exactly two ranks for the two-node Jacobi example\n";
        }
        return 1;
    }

    constexpr std::size_t nx = 64;
    constexpr std::size_t ny = 34;
    constexpr int iterations = 4;

    mkdir("outputs", 0775);
    mkdir("outputs/sim_traces", 0775);
    const std::string timestamp = run_timestamp();
    const std::string trace_path =
        "outputs/sim_traces/routing_jacobi_mpi_node" +
        std::to_string(runtime.node_id()) + "_sim_jobs_" + timestamp + ".jsonl";
    const std::string log_path =
        "outputs/routing_jacobi_mpi_node" +
        std::to_string(runtime.node_id()) + "_run_log_" + timestamp + ".jsonl";
    std::ofstream(trace_path, std::ios::trunc).close();
    std::ofstream(log_path, std::ios::trunc).close();
    setenv("ROUTING_SIM_TRACE", trace_path.c_str(), 1);
    setenv("ROUTING_RUN_LOG", log_path.c_str(), 1);

    const std::vector<float> initial_global = make_initial_grid(nx, ny);
    const std::size_t interior_rows = ny - 2;
    const std::size_t node0_rows = interior_rows / 2;
    const std::size_t node1_rows = interior_rows - node0_rows;

    LogicalNode node;
    node.node_id = runtime.node_id();
    node.neighbor_node_id = node.node_id == 0 ? 1 : 0;
    node.start_global_row = node.node_id == 0 ? 1 : 1 + node0_rows;
    node.owned_rows = node.node_id == 0 ? node0_rows : node1_rows;
    node.nx = nx;
    const bool node0_gpu_available =
        node.node_id == 0 && env_enabled("ROUTING_ENABLE_NODE0_GPU");
    node.router_config = make_logical_node_config(
        node.node_id,
        node0_gpu_available);
    load_node_from_global(node, initial_global, ny);

    std::size_t next_job_id =
        static_cast<std::size_t>(runtime.node_id()) * 2 + 1;
    for (int iter = 0; iter < iterations; ++iter) {
        route_iteration(runtime, node, iter, next_job_id);
        next_job_id += 2;
    }

    const std::vector<float> actual = runtime.gather_jacobi_global(
        *node.input,
        nx,
        node.owned_rows,
        node.start_global_row,
        ny,
        initial_global);

    if (runtime.rank() == 0) {
        const std::vector<float> expected =
            run_reference(initial_global, nx, ny, iterations);
        if (!verify(actual, expected)) {
            std::cerr << "MPI two-logical-node Jacobi verification failed\n";
            return 1;
        }

        std::cout << "MPI two-logical-node Jacobi verification passed\n";
        std::cout << "Logical node 0 endpoints: CPU0, "
                  << (env_enabled("ROUTING_ENABLE_NODE0_GPU") ? "GPU0, " : "")
                  << "SIM0\n";
        std::cout << "Logical node 1 endpoints: CPU1, SIM1\n";
    }

    std::cout << "Node " << runtime.node_id()
              << " SIM trace emitted to " << trace_path << '\n';
    std::cout << "Node " << runtime.node_id()
              << " run log emitted to " << log_path << '\n';
    return 0;
}
