#include "kernels.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace Routing
{
namespace
{
using Clock = std::chrono::steady_clock;

double sim_arrival_us()
{
    // Arrival is measured at SIM operator entry: the moment the runtime has
    // routed a ready job to the simulated endpoint.
    static const Clock::time_point start = Clock::now();
    const auto elapsed = Clock::now() - start;
    return std::chrono::duration<double, std::micro>(elapsed).count();
}

const char* sim_trace_path()
{
    const char* configured = std::getenv("ROUTING_SIM_TRACE");
    return configured != nullptr ? configured : "outputs/sim_traces/sim_jobs.jsonl";
}

void ensure_trace_dirs()
{
    mkdir("outputs", 0775);
    mkdir("outputs/sim_traces", 0775);
}

std::ofstream open_trace()
{
    // SIM traces are inputs to the offline SimPy model. The C++ runtime only
    // records what arrived; Python computes modeled timing later.
    ensure_trace_dirs();
    std::ofstream trace(sim_trace_path(), std::ios::app);
    if (!trace) {
        throw std::runtime_error("Failed to open SIM trace output");
    }
    return trace;
}

void emit_common_prefix(
    std::ofstream& trace,
    const JobMetadata& metadata,
    const char* op,
    double arrival_us)
{
    trace << "{\"job_id\":" << metadata.job_id
          << ",\"node_id\":" << metadata.node_id
          << ",\"endpoint\":\"SIM" << metadata.node_id << "\""
          << ",\"op\":\"" << op << "\""
          << ",\"arrival_us\":" << arrival_us;
}

void emit_saxpy_trace(std::size_t n, const JobMetadata& metadata, double arrival_us)
{
    std::ofstream trace = open_trace();
    emit_common_prefix(trace, metadata, "SAXPY", arrival_us);
    trace << ",\"n\":" << n
          << ",\"input_bytes\":" << n * sizeof(float) * 2
          << ",\"output_bytes\":" << n * sizeof(float)
          << ",\"metadata\":{\"dtype\":\"float32\",\"layout\":\"contiguous\"}"
          << "}\n";
}

std::size_t jacobi_work_units(
    const char* op,
    std::size_t nx,
    std::size_t ny,
    std::size_t halo_width)
{
    // Keep trace byte counts aligned with the router's cost metrics: interior
    // jobs use full-grid buffers, while boundary jobs use boundary-sized work.
    const std::string op_name(op);
    if (op_name == "JACOBI_INTERIOR") {
        const std::size_t width = nx > 2 * halo_width ?
            nx - 2 * halo_width : 0;
        const std::size_t height = ny > 2 * halo_width ?
            ny - 2 * halo_width : 0;
        return width * height;
    }

    const std::size_t horizontal = std::min(2 * halo_width, ny) * nx;
    const std::size_t vertical =
        (ny > 2 * halo_width ? ny - 2 * halo_width : 0) *
        std::min(2 * halo_width, nx);
    return horizontal + vertical;
}

void emit_jacobi_trace(
    const char* op,
    std::size_t nx,
    std::size_t ny,
    std::size_t halo_width,
    const JobMetadata& metadata,
    double arrival_us)
{
    const std::size_t work_units = jacobi_work_units(op, nx, ny, halo_width);
    const bool is_interior = std::string(op) == "JACOBI_INTERIOR";
    const std::size_t input_bytes = is_interior ?
        nx * ny * sizeof(float) : work_units * sizeof(float) * 2;
    const std::size_t output_bytes = is_interior ?
        nx * ny * sizeof(float) : work_units * sizeof(float);

    std::ofstream trace = open_trace();
    emit_common_prefix(trace, metadata, op, arrival_us);
    trace << ",\"nx\":" << nx
          << ",\"ny\":" << ny
          << ",\"halo_width\":" << halo_width
          << ",\"input_bytes\":" << input_bytes
          << ",\"output_bytes\":" << output_bytes
          << ",\"metadata\":{\"stencil\":\"2d_5pt\",\"dtype\":\"float32\","
          << "\"layout\":\"row_major\",\"iteration\":" << metadata.iteration
          << ",\"neighbor_node_id\":" << metadata.neighbor_node_id << "}"
          << "}\n";
}
}

void saxpy_simpy(
    float a,
    std::size_t n,
    const float* x,
    float* y,
    const JobMetadata& metadata)
{
    const double arrival_us = sim_arrival_us();
    emit_saxpy_trace(n, metadata, arrival_us);
    // SIM models performance only; CPU fallback performs the actual math so
    // ForceSim remains numerically correct in examples.
    saxpy_cpu_only(a, n, x, y);
}

void jacobi_interior_simpy(
    std::size_t nx,
    std::size_t ny,
    std::size_t halo_width,
    const float* input,
    float* output,
    const JobMetadata& metadata)
{
    const double arrival_us = sim_arrival_us();
    emit_jacobi_trace("JACOBI_INTERIOR", nx, ny, halo_width, metadata, arrival_us);
    jacobi_interior_cpu(nx, ny, halo_width, input, output);
}

void jacobi_boundary_simpy(
    std::size_t nx,
    std::size_t ny,
    std::size_t halo_width,
    const float* input,
    float* output,
    const JobMetadata& metadata)
{
    const double arrival_us = sim_arrival_us();
    emit_jacobi_trace("JACOBI_BOUNDARY", nx, ny, halo_width, metadata, arrival_us);
    jacobi_boundary_cpu(nx, ny, halo_width, input, output);
}

void jacobi_halo_boundary_simpy(
    std::size_t nx,
    std::size_t ny,
    std::size_t halo_width,
    const float* input,
    float* output,
    const JobMetadata& metadata)
{
    const double arrival_us = sim_arrival_us();
    emit_jacobi_trace("JACOBI_HALO_BOUNDARY", nx, ny, halo_width, metadata, arrival_us);
    jacobi_boundary_cpu(nx, ny, halo_width, input, output);
}
}
