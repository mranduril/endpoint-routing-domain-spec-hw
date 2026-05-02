#include "router.h"

#include <algorithm>
#include <stdexcept>
#include <variant>

namespace Routing
{
DispatchPlan make_cpu_plan(std::size_t n, double estimated_cost)
{
    DispatchPlan plan{};
    plan.kind = DispatchKind::CpuOnly;
    plan.node_kind = NodeKind::Local;
    plan.cpu_begin = 0;
    plan.cpu_end = n;
    plan.gpu_begin = 0;
    plan.gpu_end = 0;
    plan.estimated_cost = estimated_cost;
    return plan;
}

DispatchPlan make_gpu_plan(std::size_t n, double estimated_cost)
{
    DispatchPlan plan{};
    plan.kind = DispatchKind::GpuOnly;
    plan.node_kind = NodeKind::Local;
    plan.cpu_begin = 0;
    plan.cpu_end = 0;
    plan.gpu_begin = 0;
    plan.gpu_end = n;
    plan.estimated_cost = estimated_cost;
    return plan;
}

DispatchPlan make_split_plan(std::size_t n, double estimated_cost)
{
    const std::size_t split = n / 2;
    DispatchPlan plan{};
    plan.kind = DispatchKind::CpuGpuSplit;
    plan.node_kind = NodeKind::Local;
    plan.cpu_begin = 0;
    plan.cpu_end = split;
    plan.gpu_begin = split;
    plan.gpu_end = n;
    plan.estimated_cost = estimated_cost;
    return plan;
}

DispatchPlan make_sim_plan(std::size_t n, double estimated_cost)
{
    DispatchPlan plan{};
    plan.kind = DispatchKind::SimOnly;
    plan.node_kind = NodeKind::Local;
    plan.cpu_begin = 0;
    plan.cpu_end = 0;
    plan.gpu_begin = 0;
    plan.gpu_end = 0;
    plan.sim_begin = 0;
    plan.sim_end = n;
    plan.estimated_cost = estimated_cost;
    return plan;
}

DispatchPlan annotate_costs(
    DispatchPlan plan,
    double cpu_cost,
    double gpu_cost,
    double split_cost,
    double sim_cost,
    std::size_t work_units)
{
    plan.cpu_estimated_cost = cpu_cost;
    plan.gpu_estimated_cost = gpu_cost;
    plan.split_estimated_cost = split_cost;
    plan.sim_estimated_cost = sim_cost;
    plan.work_units = work_units;
    return plan;
}

double estimate_split_cost(const payloadSAXPY& payload, const RouterConfig& config)
{
    const std::size_t cpu_work = payload.n / 2;
    const std::size_t gpu_work = payload.n - cpu_work;
    constexpr std::size_t saxpy_transfer_copies = 3; // x H2D, y H2D, y D2H

    const double cpu_cost = config.cpu_alpha * static_cast<double>(cpu_work);
    const double gpu_cost = config.gpu_launch +
        config.gpu_alpha * static_cast<double>(gpu_work) +
        config.gpu_transfer_alpha * static_cast<double>(gpu_work) *
            (sizeof(float) * saxpy_transfer_copies);

    // Simple overlap model: both endpoints work in parallel on their slices.
    return std::max(cpu_cost, gpu_cost);
}

std::size_t jacobi_work_units(const payloadJacobi& payload, WorkloadType type)
{
    if (type == WorkloadType::JACOBI_INTERIOR) {
        const std::size_t width = payload.nx > 2 * payload.halo_width ?
            payload.nx - 2 * payload.halo_width : 0;
        const std::size_t height = payload.ny > 2 * payload.halo_width ?
            payload.ny - 2 * payload.halo_width : 0;
        return width * height;
    }

    const std::size_t horizontal =
        std::min(2 * payload.halo_width, payload.ny) * payload.nx;
    const std::size_t vertical =
        (payload.ny > 2 * payload.halo_width ?
            payload.ny - 2 * payload.halo_width : 0) *
        std::min(2 * payload.halo_width, payload.nx);
    return horizontal + vertical;
}

Router::Router(RouterConfig config)
    : config_(config)
{
}

DispatchPlan Router::plan(const Job& job, RoutingPolicy policy) const
{
    if (!job.validate()) {
        throw std::invalid_argument("Router received an invalid job");
    }

    switch (job.type) {
        case WorkloadType::SAXPY:
            if (!std::holds_alternative<payloadSAXPY>(job.payload)) {
                throw std::invalid_argument("Job payload does not match SAXPY workload");
            }
            return plan_saxpy(std::get<payloadSAXPY>(job.payload), policy);
        case WorkloadType::JACOBI_INTERIOR:
        case WorkloadType::JACOBI_BOUNDARY:
        case WorkloadType::JACOBI_HALO_BOUNDARY:
            if (!std::holds_alternative<payloadJacobi>(job.payload)) {
                throw std::invalid_argument("Job payload does not match Jacobi workload");
            }
            return plan_jacobi(std::get<payloadJacobi>(job.payload), job.type, policy);
        default:
            throw std::invalid_argument("Router does not support this workload yet");
    }
}

DispatchPlan Router::plan_jacobi(
    const payloadJacobi& payload,
    WorkloadType type,
    RoutingPolicy policy) const
{
    const std::size_t work_units = jacobi_work_units(payload, type);
    const double cpu_cost = estimate_cpu(payload);
    const double gpu_cost = estimate_gpu(payload);
    const double sim_cost = estimate_sim(payload);
    const double split_cost = std::max(cpu_cost, gpu_cost);

    switch (policy) {
        case RoutingPolicy::ForceCpu:
            return annotate_costs(
                make_cpu_plan(work_units, cpu_cost),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                work_units);
        case RoutingPolicy::ForceGpu:
            return annotate_costs(
                make_gpu_plan(work_units, gpu_cost),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                work_units);
        case RoutingPolicy::ForceSim:
            return annotate_costs(
                make_sim_plan(work_units, sim_cost),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                work_units);
        case RoutingPolicy::ForceSplit:
            return annotate_costs(
                make_split_plan(work_units, split_cost),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                work_units);
        case RoutingPolicy::Auto:
        default: {
            DispatchPlan best_plan = make_cpu_plan(work_units, cpu_cost);

            // Boundary work is the useful SIM endpoint story; interior work can use GPU.
            if (type == WorkloadType::JACOBI_INTERIOR &&
                gpu_cost < best_plan.estimated_cost) {
                best_plan = make_gpu_plan(work_units, gpu_cost);
            }

            if (type != WorkloadType::JACOBI_INTERIOR &&
                sim_cost < best_plan.estimated_cost) {
                best_plan = make_sim_plan(work_units, sim_cost);
            }

            return annotate_costs(
                best_plan,
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                work_units);
        }
    }
}

DispatchPlan Router::plan_saxpy(const payloadSAXPY& payload, RoutingPolicy policy) const
{
    const double cpu_cost = estimate_cpu(payload);
    const double gpu_cost = estimate_gpu(payload);
    const double sim_cost = estimate_sim(payload);
    const double split_cost = estimate_split_cost(payload, config_);

    switch (policy) {
        case RoutingPolicy::ForceCpu:
            return annotate_costs(
                make_cpu_plan(payload.n, cpu_cost),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                payload.n);
        case RoutingPolicy::ForceGpu:
            return annotate_costs(
                make_gpu_plan(payload.n, gpu_cost),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                payload.n);
        case RoutingPolicy::ForceSplit:
            return annotate_costs(
                make_split_plan(payload.n, split_cost),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                payload.n);
        case RoutingPolicy::ForceSim:
            return annotate_costs(
                make_sim_plan(payload.n, sim_cost),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                payload.n);
        case RoutingPolicy::Auto:
        default: {
            DispatchPlan best_plan = make_cpu_plan(payload.n, cpu_cost);

            if (gpu_cost < best_plan.estimated_cost) {
                best_plan = make_gpu_plan(payload.n, gpu_cost);
            }

            if (split_cost < best_plan.estimated_cost) {
                best_plan = make_split_plan(payload.n, split_cost);
            }

            if (sim_cost < best_plan.estimated_cost) {
                best_plan = make_sim_plan(payload.n, sim_cost);
            }

            return annotate_costs(
                best_plan,
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                payload.n);
        }
    }
}

double Router::estimate_cpu(const payloadSAXPY& payload) const
{
    return config_.cpu_alpha * static_cast<double>(payload.n);
}

double Router::estimate_gpu(const payloadSAXPY& payload) const
{
    return config_.gpu_launch
        + config_.gpu_alpha * static_cast<double>(payload.n)
        + config_.gpu_transfer_alpha * static_cast<double>(payload.n) * (sizeof(float) * 3); // Account for 2 h2d and 1 d2h transfers
}

double Router::estimate_sim(const payloadSAXPY& payload) const
{
    return config_.sim_startup +
        config_.sim_alpha * static_cast<double>(payload.n);
}

double Router::estimate_cpu(const payloadJacobi& payload) const
{
    return config_.cpu_alpha * static_cast<double>(payload.nx * payload.ny);
}

double Router::estimate_gpu(const payloadJacobi& payload) const
{
    constexpr std::size_t input_output_copies = 2;
    return config_.gpu_launch
        + config_.gpu_alpha * static_cast<double>(payload.nx * payload.ny)
        + config_.gpu_transfer_alpha * static_cast<double>(payload.nx * payload.ny) *
            (sizeof(float) * input_output_copies);
}

double Router::estimate_sim(const payloadJacobi& payload) const
{
    const std::size_t cells = payload.nx * payload.ny;
    return config_.sim_startup + config_.sim_alpha * static_cast<double>(cells);
}

} // namespace Routing
