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
    plan.cpu_begin = 0;
    plan.cpu_end = 0;
    plan.gpu_begin = 0;
    plan.gpu_end = 0;
    plan.sim_begin = 0;
    plan.sim_end = n;
    plan.estimated_cost = estimated_cost;
    return plan;
}

double estimate_split_cost(const payloadSAXPY& payload, const RouterConfig& config)
{
    const std::size_t cpu_work = payload.n / 2;
    const std::size_t gpu_work = payload.n - cpu_work;

    const double cpu_cost = config.cpu_alpha * static_cast<double>(cpu_work);
    const double gpu_cost = config.gpu_launch +
        config.gpu_alpha * static_cast<double>(gpu_work);

    // Simple overlap model: both endpoints work in parallel on their slices.
    return std::max(cpu_cost, gpu_cost);
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
        default:
            throw std::invalid_argument("Router does not support this workload yet");
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
            return make_cpu_plan(payload.n, cpu_cost);
        case RoutingPolicy::ForceGpu:
            return make_gpu_plan(payload.n, gpu_cost);
        case RoutingPolicy::ForceSplit:
            return make_split_plan(payload.n, split_cost);
        case RoutingPolicy::ForceSim:
            return make_sim_plan(payload.n, sim_cost);
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

            return best_plan;
        }
    }
}

double Router::estimate_cpu(const payloadSAXPY& payload) const
{
    return config_.cpu_alpha * static_cast<double>(payload.n);
}

double Router::estimate_gpu(const payloadSAXPY& payload) const
{
    return config_.gpu_launch +
        config_.gpu_alpha * static_cast<double>(payload.n);
}

double Router::estimate_sim(const payloadSAXPY& payload) const
{
    return config_.sim_startup +
        config_.sim_alpha * static_cast<double>(payload.n);
}

} // namespace Routing
