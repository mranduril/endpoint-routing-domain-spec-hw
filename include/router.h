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

struct DispatchPlan {
    NodeKind node_kind;
    DispatchKind kind;
    std::size_t cpu_begin = 0;
    std::size_t cpu_end = 0;
    std::size_t gpu_begin = 0;
    std::size_t gpu_end = 0;
    std::size_t sim_begin = 0;
    std::size_t sim_end = 0;
    double estimated_cost = 0.0;
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

struct RouterConfig {
    double cpu_alpha = 1.0;
    double gpu_launch = 1000.0;
    double gpu_alpha = 0.2;
    double gpu_transfer_alpha = 0.5;
    double sim_startup = 5000.0;
    double sim_alpha = 2.0;
};

class Router {
public:
    explicit Router(RouterConfig config = {});
    DispatchPlan plan(const Job& job,
                      RoutingPolicy policy = RoutingPolicy::Auto) const;

private:
    RouterConfig config_;

    DispatchPlan plan_saxpy(const payloadSAXPY& payload,
                            RoutingPolicy policy) const;

    double estimate_cpu(const payloadSAXPY& payload) const;
    double estimate_gpu(const payloadSAXPY& payload) const;
    double estimate_sim(const payloadSAXPY& payload) const;
};

} // namespace Routing

#endif // ROUTER_H