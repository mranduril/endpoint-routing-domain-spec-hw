#include "router.h"

#include <algorithm>
#include <stdexcept>
#include <variant>

namespace Routing
{
NodeKind node_kind_for(const JobMetadata& metadata, const RouterConfig& config)
{
    return metadata.node_id == config.local_node_id ?
        NodeKind::Local : NodeKind::Remote;
}

void annotate_node(DispatchPlan& plan, const JobMetadata& metadata, const RouterConfig& config)
{
    // "Remote" is only a planning/logging label today. The runtime still needs
    // an MPI executor before remote plans can execute on another process.
    plan.target_node_id = metadata.node_id;
    plan.node_kind = node_kind_for(metadata, config);
}

DispatchPlan make_cpu_plan(
    std::size_t n,
    double estimated_cost,
    const JobMetadata& metadata,
    const RouterConfig& config)
{
    DispatchPlan plan{};
    plan.kind = DispatchKind::CpuOnly;
    annotate_node(plan, metadata, config);
    plan.cpu_begin = 0;
    plan.cpu_end = n;
    plan.gpu_begin = 0;
    plan.gpu_end = 0;
    plan.estimated_cost = estimated_cost;
    return plan;
}

DispatchPlan make_gpu_plan(
    std::size_t n,
    double estimated_cost,
    const JobMetadata& metadata,
    const RouterConfig& config)
{
    DispatchPlan plan{};
    plan.kind = DispatchKind::GpuOnly;
    annotate_node(plan, metadata, config);
    plan.cpu_begin = 0;
    plan.cpu_end = 0;
    plan.gpu_begin = 0;
    plan.gpu_end = n;
    plan.estimated_cost = estimated_cost;
    return plan;
}

DispatchPlan make_split_plan(
    std::size_t n,
    double estimated_cost,
    const JobMetadata& metadata,
    const RouterConfig& config)
{
    const std::size_t split = n / 2;
    DispatchPlan plan{};
    plan.kind = DispatchKind::CpuGpuSplit;
    annotate_node(plan, metadata, config);
    plan.cpu_begin = 0;
    plan.cpu_end = split;
    plan.gpu_begin = split;
    plan.gpu_end = n;
    plan.estimated_cost = estimated_cost;
    return plan;
}

DispatchPlan make_sim_plan(
    std::size_t n,
    double estimated_cost,
    const JobMetadata& metadata,
    const RouterConfig& config)
{
    DispatchPlan plan{};
    plan.kind = DispatchKind::SimOnly;
    annotate_node(plan, metadata, config);
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
    double remote_cost,
    const Router::JobCostMetrics& metrics)
{
    plan.cpu_estimated_cost = cpu_cost;
    plan.gpu_estimated_cost = gpu_cost;
    plan.split_estimated_cost = split_cost;
    plan.sim_estimated_cost = sim_cost;
    plan.remote_estimated_cost = remote_cost;
    plan.work_units = metrics.work_units;
    plan.input_bytes = metrics.input_bytes;
    plan.output_bytes = metrics.output_bytes;
    return plan;
}

double effective_cpu_queue(const RouterConfig& config)
{
    // Runtime queue pressure is intentionally simple: each in-flight job adds
    // an endpoint-specific penalty to the configured base queue estimate.
    return config.queue_cpu +
        config.cpu_queue_job_penalty * static_cast<double>(config.queued_cpu_jobs);
}

double effective_gpu_queue(const RouterConfig& config)
{
    return config.queue_gpu +
        config.gpu_queue_job_penalty * static_cast<double>(config.queued_gpu_jobs);
}

double effective_sim_queue(const RouterConfig& config)
{
    return config.queue_sim +
        config.sim_queue_job_penalty * static_cast<double>(config.queued_sim_jobs);
}

double host_transfer_penalty(
    const Router::JobCostMetrics& metrics,
    const RouterConfig& config)
{
    // CPU execution assumes host-resident inputs and host-visible outputs. If
    // the job metadata says otherwise, charge the configured movement cost.
    const std::size_t input_bytes =
        metrics.input_location == DataLocation::Host ? 0 : metrics.input_bytes;
    const std::size_t output_bytes =
        metrics.output_location == DataLocation::Host ? 0 : metrics.output_bytes;
    return config.move_to_host_per_byte *
        static_cast<double>(input_bytes + output_bytes);
}

double gpu_transfer_in_penalty(
    const Router::JobCostMetrics& metrics,
    const RouterConfig& config)
{
    // Avoid charging H2D/D2H copies when the job declares data already resident
    // on the GPU or wants the result to remain there.
    return metrics.input_location == DataLocation::GPU ?
        0.0 :
        config.copy_to_gpu_per_byte * static_cast<double>(metrics.input_bytes);
}

double gpu_transfer_out_penalty(
    const Router::JobCostMetrics& metrics,
    const RouterConfig& config)
{
    return metrics.output_location == DataLocation::GPU ?
        0.0 :
        config.copy_back_per_byte * static_cast<double>(metrics.output_bytes);
}

double sim_transfer_in_penalty(
    const Router::JobCostMetrics& metrics,
    const RouterConfig& config)
{
    // SIM represents a modeled service endpoint, so transfers are modeled unless
    // metadata explicitly says the relevant data already lives at the SIM side.
    return metrics.input_location == DataLocation::SIM ?
        0.0 :
        config.sim_transfer_in_per_byte * static_cast<double>(metrics.input_bytes);
}

double sim_transfer_out_penalty(
    const Router::JobCostMetrics& metrics,
    const RouterConfig& config)
{
    return metrics.output_location == DataLocation::SIM ?
        0.0 :
        config.sim_transfer_out_per_byte * static_cast<double>(metrics.output_bytes);
}

double estimate_cpu_slice(
    std::size_t work_units,
    double host_move_penalty,
    const RouterConfig& config)
{
    return effective_cpu_queue(config) +
        config.cpu_fixed +
        config.cpu_per_work_unit * static_cast<double>(work_units) +
        host_move_penalty;
}

double estimate_gpu_slice(
    std::size_t work_units,
    double transfer_in_penalty,
    double transfer_out_penalty,
    const RouterConfig& config)
{
    return effective_gpu_queue(config) +
        transfer_in_penalty +
        config.cuda_launch +
        config.gpu_per_work_unit * static_cast<double>(work_units) +
        transfer_out_penalty;
}

double estimate_split_cost(
    const Router::JobCostMetrics& metrics,
    const RouterConfig& config)
{
    // Split plans model CPU and GPU slices running concurrently, so elapsed cost
    // is max(slice costs), not their sum.
    const std::size_t cpu_work = metrics.work_units / 2;
    const std::size_t gpu_work = metrics.work_units - cpu_work;
    Router::JobCostMetrics cpu_metrics = metrics;
    Router::JobCostMetrics gpu_metrics = metrics;
    cpu_metrics.work_units = cpu_work;
    gpu_metrics.work_units = gpu_work;
    cpu_metrics.input_bytes = metrics.input_bytes / 2;
    cpu_metrics.output_bytes = metrics.output_bytes / 2;
    gpu_metrics.input_bytes = metrics.input_bytes - cpu_metrics.input_bytes;
    gpu_metrics.output_bytes = metrics.output_bytes - cpu_metrics.output_bytes;

    const double cpu_cost = estimate_cpu_slice(
        cpu_work,
        host_transfer_penalty(cpu_metrics, config),
        config);
    const double gpu_cost = estimate_gpu_slice(
        gpu_work,
        gpu_transfer_in_penalty(gpu_metrics, config),
        gpu_transfer_out_penalty(gpu_metrics, config),
        config);
    // Simple overlap model: both endpoints work in parallel on their slices.
    return std::max(cpu_cost, gpu_cost);
}

Router::JobCostMetrics saxpy_metrics(const payloadSAXPY& payload)
{
    // SAXPY reads x and y, then writes y. Count that as two input vectors and
    // one output vector for movement/modeling purposes.
    Router::JobCostMetrics metrics;
    metrics.work_units = payload.n;
    metrics.input_bytes = payload.n * sizeof(float) * 2;
    metrics.output_bytes = payload.n * sizeof(float);
    return metrics;
}

void apply_locations(Router::JobCostMetrics& metrics, const JobMetadata& metadata)
{
    // Keep work/byte metrics independent from data residency so tests and logs
    // can explain both the job shape and movement penalties.
    metrics.input_location = metadata.input_location;
    metrics.output_location = metadata.output_location;
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

Router::JobCostMetrics jacobi_metrics(const payloadJacobi& payload, WorkloadType type)
{
    Router::JobCostMetrics metrics;
    metrics.work_units = jacobi_work_units(payload, type);

    if (type == WorkloadType::JACOBI_INTERIOR) {
        // Interior kernels currently consume/copy full grids in the concrete
        // CPU/GPU/SIM wrappers, so model full-grid movement.
        metrics.input_bytes = payload.nx * payload.ny * sizeof(float);
        metrics.output_bytes = payload.nx * payload.ny * sizeof(float);
    } else {
        // Boundary and halo-boundary jobs are modeled as boundary-band work so
        // they do not pay full-grid movement in the router or SimPy trace.
        metrics.input_bytes = metrics.work_units * sizeof(float) * 2;
        metrics.output_bytes = metrics.work_units * sizeof(float);
    }

    return metrics;
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
            return plan_saxpy(
                std::get<payloadSAXPY>(job.payload),
                job.metadata,
                policy);
        case WorkloadType::JACOBI_INTERIOR:
        case WorkloadType::JACOBI_BOUNDARY:
        case WorkloadType::JACOBI_HALO_BOUNDARY:
            if (!std::holds_alternative<payloadJacobi>(job.payload)) {
                throw std::invalid_argument("Job payload does not match Jacobi workload");
            }
            return plan_jacobi(
                std::get<payloadJacobi>(job.payload),
                job.type,
                job.metadata,
                policy);
        default:
            throw std::invalid_argument("Router does not support this workload yet");
    }
}

DispatchPlan Router::plan_jacobi(
    const payloadJacobi& payload,
    WorkloadType type,
    const JobMetadata& metadata,
    RoutingPolicy policy) const
{
    const JobCostMetrics metrics = jacobi_metrics(payload, type);
    JobCostMetrics located_metrics = metrics;
    apply_locations(located_metrics, metadata);
    const double remote_cost = estimate_remote(located_metrics, metadata);
    const double cpu_cost = estimate_cpu(located_metrics) + remote_cost;
    const double gpu_cost = estimate_gpu(located_metrics) + remote_cost;
    const double sim_cost = estimate_sim(located_metrics) + remote_cost;
    const double split_cost = estimate_split_cost(located_metrics, config_) + remote_cost;

    switch (policy) {
        case RoutingPolicy::ForceCpu:
            return annotate_costs(
                make_cpu_plan(metrics.work_units, cpu_cost, metadata, config_),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                remote_cost,
                located_metrics);
        case RoutingPolicy::ForceGpu:
            return annotate_costs(
                make_gpu_plan(metrics.work_units, gpu_cost, metadata, config_),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                remote_cost,
                located_metrics);
        case RoutingPolicy::ForceSim:
            return annotate_costs(
                make_sim_plan(metrics.work_units, sim_cost, metadata, config_),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                remote_cost,
                located_metrics);
        case RoutingPolicy::ForceSplit:
            return annotate_costs(
                make_split_plan(metrics.work_units, split_cost, metadata, config_),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                remote_cost,
                located_metrics);
        case RoutingPolicy::Auto:
        default: {
            DispatchPlan best_plan = make_cpu_plan(
                metrics.work_units,
                cpu_cost,
                metadata,
                config_);

            // Boundary work is the useful SIM endpoint story; interior work can use GPU.
            if (type == WorkloadType::JACOBI_INTERIOR &&
                gpu_cost < best_plan.estimated_cost) {
                best_plan = make_gpu_plan(metrics.work_units, gpu_cost, metadata, config_);
            }

            if (type != WorkloadType::JACOBI_INTERIOR &&
                sim_cost < best_plan.estimated_cost) {
                best_plan = make_sim_plan(metrics.work_units, sim_cost, metadata, config_);
            }

            return annotate_costs(
                best_plan,
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                remote_cost,
                located_metrics);
        }
    }
}

DispatchPlan Router::plan_saxpy(
    const payloadSAXPY& payload,
    const JobMetadata& metadata,
    RoutingPolicy policy) const
{
    const JobCostMetrics metrics = saxpy_metrics(payload);
    JobCostMetrics located_metrics = metrics;
    apply_locations(located_metrics, metadata);
    const double remote_cost = estimate_remote(located_metrics, metadata);
    const double cpu_cost = estimate_cpu(located_metrics) + remote_cost;
    const double gpu_cost = estimate_gpu(located_metrics) + remote_cost;
    const double sim_cost = estimate_sim(located_metrics) + remote_cost;
    const double split_cost = estimate_split_cost(located_metrics, config_) + remote_cost;

    switch (policy) {
        case RoutingPolicy::ForceCpu:
            return annotate_costs(
                make_cpu_plan(metrics.work_units, cpu_cost, metadata, config_),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                remote_cost,
                located_metrics);
        case RoutingPolicy::ForceGpu:
            return annotate_costs(
                make_gpu_plan(metrics.work_units, gpu_cost, metadata, config_),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                remote_cost,
                located_metrics);
        case RoutingPolicy::ForceSplit:
            return annotate_costs(
                make_split_plan(metrics.work_units, split_cost, metadata, config_),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                remote_cost,
                located_metrics);
        case RoutingPolicy::ForceSim:
            return annotate_costs(
                make_sim_plan(metrics.work_units, sim_cost, metadata, config_),
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                remote_cost,
                located_metrics);
        case RoutingPolicy::Auto:
        default: {
            DispatchPlan best_plan = make_cpu_plan(
                metrics.work_units,
                cpu_cost,
                metadata,
                config_);

            if (gpu_cost < best_plan.estimated_cost) {
                best_plan = make_gpu_plan(metrics.work_units, gpu_cost, metadata, config_);
            }

            if (split_cost < best_plan.estimated_cost) {
                best_plan = make_split_plan(metrics.work_units, split_cost, metadata, config_);
            }

            if (sim_cost < best_plan.estimated_cost) {
                best_plan = make_sim_plan(metrics.work_units, sim_cost, metadata, config_);
            }

            return annotate_costs(
                best_plan,
                cpu_cost,
                gpu_cost,
                split_cost,
                sim_cost,
                remote_cost,
                located_metrics);
        }
    }
}

double Router::estimate_cpu(const JobCostMetrics& metrics) const
{
    return estimate_cpu_slice(
        metrics.work_units,
        host_transfer_penalty(metrics, config_),
        config_);
}

double Router::estimate_gpu(const JobCostMetrics& metrics) const
{
    return estimate_gpu_slice(
        metrics.work_units,
        gpu_transfer_in_penalty(metrics, config_),
        gpu_transfer_out_penalty(metrics, config_),
        config_);
}

double Router::estimate_sim(const JobCostMetrics& metrics) const
{
    return effective_sim_queue(config_) +
        sim_transfer_in_penalty(metrics, config_) +
        config_.sim_setup +
        config_.sim_per_work_unit * static_cast<double>(metrics.work_units) +
        sim_transfer_out_penalty(metrics, config_);
}

double Router::estimate_remote(
    const JobCostMetrics& metrics,
    const JobMetadata& metadata) const
{
    if (metadata.node_id == config_.local_node_id) {
        return 0.0;
    }

    return config_.remote_fixed +
        config_.remote_transfer_per_byte *
            static_cast<double>(metrics.input_bytes + metrics.output_bytes);
}

} // namespace Routing
