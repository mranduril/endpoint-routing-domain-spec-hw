#ifndef ROUTER_H
#define ROUTER_H

#include "job.h"
#include "workload_type.h"
#include <cstddef>
#include <unordered_map>
#include <string>

namespace Routing {

enum class DispatchKind {
    CpuOnly,
    GpuOnly,
    CpuGpuSplit,
    SimOnly
};

enum class NodeKind {
    Local,
    Remote
};

enum class EndpointKind {
    CPU,
    GPU,
    SIM
};

struct EndpointAvailability {
    bool cpu = true;
    bool gpu = true;
    bool sim = true;
};

struct DispatchPlan {
    // The plan is both an execution recipe and a logging artifact. The ranges
    // describe the logical work slice assigned to each endpoint.
    NodeKind node_kind;
    DispatchKind kind;
    int target_node_id = 0;
    std::size_t cpu_begin = 0;
    std::size_t cpu_end = 0;
    std::size_t gpu_begin = 0;
    std::size_t gpu_end = 0;
    std::size_t sim_begin = 0;
    std::size_t sim_end = 0;
    double estimated_cost = 0.0;
    double cpu_estimated_cost = 0.0;
    double gpu_estimated_cost = 0.0;
    double split_estimated_cost = 0.0;
    double sim_estimated_cost = 0.0;
    double remote_estimated_cost = 0.0;
    std::size_t work_units = 0;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
};

enum class RoutingPolicy {
    Auto,
    ForceCpu,
    ForceGpu,
    ForceSplit,
    ForceSim
};

inline std::unordered_map<RoutingPolicy, std::string> RoutingPolicyNames = {
    {RoutingPolicy::Auto, "Auto"},
    {RoutingPolicy::ForceCpu, "ForceCpu"},
    {RoutingPolicy::ForceGpu, "ForceGpu"},
    {RoutingPolicy::ForceSplit, "ForceSplit"},
    {RoutingPolicy::ForceSim, "ForceSim"}
};

enum class CostModelPreset {
    Default,
    SimPyAlignedStencil
};

struct RouterConfig {
    int local_node_id = 0;
    std::unordered_map<int, EndpointAvailability> endpoint_availability;

    // Base queue terms can be supplied externally. The runtime also fills the
    // queued_*_jobs fields from currently in-flight endpoint work before
    // calling the router, giving the cost model lightweight queue pressure.
    double queue_cpu = 0.0;
    double queue_gpu = 0.0;
    double queue_sim = 0.0;
    std::size_t queued_cpu_jobs = 0;
    std::size_t queued_gpu_jobs = 0;
    std::size_t queued_sim_jobs = 0;
    double cpu_queue_job_penalty = 250.0;
    double gpu_queue_job_penalty = 500.0;
    double sim_queue_job_penalty = 500.0;

    // Endpoint execution terms. These implement:
    // queue + fixed/setup/launch + per-work + data movement when needed.
    double cpu_fixed = 0.0;
    double cpu_per_work_unit = 1.0;
    double move_to_host_per_byte = 0.0;
    double cuda_launch = 1000.0;
    double gpu_per_work_unit = 0.2;
    double copy_to_gpu_per_byte = 0.5;
    double copy_back_per_byte = 0.5;
    double sim_setup = 5000.0;
    double sim_per_work_unit = 2.0;
    double sim_transfer_in_per_byte = 0.5;
    double sim_transfer_out_per_byte = 0.5;

    // Structural multi-node support: remote jobs receive this extra cost and
    // are labeled Remote, but actual remote execution is a later MPI layer.
    double remote_fixed = 10000.0;
    double remote_transfer_per_byte = 1.0;
};

RouterConfig make_router_config(
    int local_node_id = 0,
    CostModelPreset preset = CostModelPreset::Default);

void apply_cost_model_preset(
    RouterConfig& config,
    CostModelPreset preset);

class Router {
public:
    struct JobCostMetrics {
        // Normalized job shape used by all endpoint estimators. This keeps the
        // formulas endpoint-centric rather than payload-specific.
        std::size_t work_units = 0;
        std::size_t input_bytes = 0;
        std::size_t output_bytes = 0;
        DataLocation input_location = DataLocation::Host;
        DataLocation output_location = DataLocation::Host;
    };

    explicit Router(RouterConfig config = {});
    DispatchPlan plan(const Job& job,
                      RoutingPolicy policy = RoutingPolicy::Auto) const;

private:
    RouterConfig config_;

    DispatchPlan plan_saxpy(const payloadSAXPY& payload,
                            const JobMetadata& metadata,
                            RoutingPolicy policy) const;
    DispatchPlan plan_jacobi(const payloadJacobi& payload,
                             WorkloadType type,
                             const JobMetadata& metadata,
                             RoutingPolicy policy) const;

    double estimate_cpu(const JobCostMetrics& metrics) const;
    double estimate_gpu(const JobCostMetrics& metrics) const;
    double estimate_sim(const JobCostMetrics& metrics) const;
    double estimate_remote(const JobCostMetrics& metrics,
                           const JobMetadata& metadata) const;
};

} // namespace Routing

#endif // ROUTER_H
