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

void exchange_halos(LogicalNode& node0, LogicalNode& node1)
{
    copy_row(*node0.input, node0.owned_rows, *node1.input, 0, node0.nx);
    copy_row(*node1.input, 1, *node0.input, node0.owned_rows + 1, node0.nx);
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
    LogicalNode& node0,
    LogicalNode& node1,
    int iteration,
    std::size_t& next_job_id)
{
    exchange_halos(node0, node1);
    *node0.output = *node0.input;
    *node1.output = *node1.input;

    Routing::Request interior0 = Routing::submit(
        make_jacobi_job(
            node0,
            Routing::WorkloadType::JACOBI_INTERIOR,
            next_job_id++,
            iteration),
        Routing::RoutingPolicy::Auto,
        node0.router_config);
    Routing::Request interior1 = Routing::submit(
        make_jacobi_job(
            node1,
            Routing::WorkloadType::JACOBI_INTERIOR,
            next_job_id++,
            iteration),
        Routing::RoutingPolicy::Auto,
        node1.router_config);
    interior0.wait();
    interior1.wait();

    Routing::Request boundary0 = Routing::submit(
        make_jacobi_job(
            node0,
            Routing::WorkloadType::JACOBI_BOUNDARY,
            next_job_id++,
            iteration),
        Routing::RoutingPolicy::ForceCpu,
        node0.router_config);
    Routing::Request boundary1 = Routing::submit(
        make_jacobi_job(
            node1,
            Routing::WorkloadType::JACOBI_BOUNDARY,
            next_job_id++,
            iteration),
        Routing::RoutingPolicy::ForceCpu,
        node1.router_config);
    boundary0.wait();
    boundary1.wait();

    node0.input.swap(node0.output);
    node1.input.swap(node1.output);
}

std::vector<float> gather_global(
    const LogicalNode& node0,
    const LogicalNode& node1,
    const std::vector<float>& initial_global,
    std::size_t nx)
{
    std::vector<float> global = initial_global;
    for (std::size_t row = 0; row < node0.owned_rows; ++row) {
        copy_row(*node0.input, row + 1, global, node0.start_global_row + row, nx);
    }
    for (std::size_t row = 0; row < node1.owned_rows; ++row) {
        copy_row(*node1.input, row + 1, global, node1.start_global_row + row, nx);
    }
    return global;
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

bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }

    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on";
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
}

int main()
{
    constexpr std::size_t nx = 64;
    constexpr std::size_t ny = 34;
    constexpr int iterations = 4;

    mkdir("outputs", 0775);
    mkdir("outputs/sim_traces", 0775);
    const std::string timestamp = run_timestamp();
    const char* trace_path = "outputs/sim_traces/routing_jacobi_two_node_sim_jobs.jsonl";
    const std::string log_path = "outputs/routing_jacobi_two_node_run_log_" +
        timestamp + ".jsonl";
    std::ofstream(trace_path, std::ios::trunc).close();
    std::ofstream(log_path, std::ios::trunc).close();
    setenv("ROUTING_SIM_TRACE", trace_path, 1);
    setenv("ROUTING_RUN_LOG", log_path.c_str(), 1);

    const std::vector<float> initial_global = make_initial_grid(nx, ny);
    const std::size_t interior_rows = ny - 2;
    const std::size_t node0_rows = interior_rows / 2;
    const std::size_t node1_rows = interior_rows - node0_rows;

    LogicalNode node0;
    node0.node_id = 0;
    node0.neighbor_node_id = 1;
    node0.start_global_row = 1;
    node0.owned_rows = node0_rows;
    node0.nx = nx;
    const bool node0_gpu_available = env_enabled("ROUTING_ENABLE_NODE0_GPU");
    node0.router_config = make_logical_node_config(0, node0_gpu_available);
    load_node_from_global(node0, initial_global, ny);

    LogicalNode node1;
    node1.node_id = 1;
    node1.neighbor_node_id = 0;
    node1.start_global_row = 1 + node0_rows;
    node1.owned_rows = node1_rows;
    node1.nx = nx;
    node1.router_config = make_logical_node_config(1, false);
    load_node_from_global(node1, initial_global, ny);

    std::size_t next_job_id = 1;
    for (int iter = 0; iter < iterations; ++iter) {
        route_iteration(node0, node1, iter, next_job_id);
    }

    const std::vector<float> actual = gather_global(node0, node1, initial_global, nx);
    const std::vector<float> expected = run_reference(initial_global, nx, ny, iterations);
    if (!verify(actual, expected)) {
        std::cerr << "Two-logical-node Jacobi verification failed\n";
        return 1;
    }

    std::cout << "Two-logical-node Jacobi verification passed\n";
    std::cout << "Logical node 0 endpoints: CPU0, "
              << (node0_gpu_available ? "GPU0, " : "")
              << "SIM0\n";
    std::cout << "Logical node 1 endpoints: CPU1, SIM1\n";
    std::cout << "SIM trace emitted to " << trace_path << '\n';
    std::cout << "Run log emitted to " << log_path << '\n';
    return 0;
}
