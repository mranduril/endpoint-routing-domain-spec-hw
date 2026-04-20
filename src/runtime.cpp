#include "runtime.h"

#include "kernels.h"

#include <exception>
#include <stdexcept>
#include <thread>
#include <variant>

namespace Routing
{

void Request::wait()
{
    if (!state_) {
        throw std::invalid_argument("Cannot wait on an invalid request");
    }

    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->cv.wait(lock, [this] {
        return state_->status == RequestStatus::Completed ||
            state_->status == RequestStatus::Failed;
    });

    if (state_->status == RequestStatus::Failed && state_->error) {
        std::rethrow_exception(state_->error);
    }
}

bool Request::ready()
{
    if (!state_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->status == RequestStatus::Completed ||
        state_->status == RequestStatus::Failed;
}

Request submit(Job job, RoutingPolicy policy)
{
    if (!job.validate()) {
        throw std::invalid_argument("Invalid job");
    }

    auto state = std::make_shared<Request::State>();
    state->job = std::move(job);

    Router router;
    state->plan = router.plan(state->job, policy);
    state->status = RequestStatus::Running;

    std::thread([state]() {
        try {
            switch (state->job.type) {
                case WorkloadType::SAXPY: {
                    if (!std::holds_alternative<payloadSAXPY>(state->job.payload)) {
                        throw std::invalid_argument("Job payload does not match SAXPY workload");
                    }

                    auto& saxpy = std::get<payloadSAXPY>(state->job.payload);
                    switch (state->plan.kind) {
                        case DispatchKind::CpuOnly:
                            saxpy_cpu_only(
                                saxpy.a,
                                saxpy.n,
                                saxpy.x->data(),
                                saxpy.y->data());
                            break;
                        default:
                            throw std::logic_error("Selected dispatch plan is not implemented yet");
                    }
                    break;
                }
                default:
                    throw std::logic_error("Runtime does not support this workload yet");
            }

            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->status = RequestStatus::Completed;
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->error = std::current_exception();
                state->status = RequestStatus::Failed;
            }
        }

        state->cv.notify_all();
    }).detach();

    return Request(std::move(state));
}

void Request::inspect_dispatch_plan() {
    std::cout << "Dispatch Plan:" << std::endl;
    std::cout << "  Node Kind: " << (state_->plan.node_kind == NodeKind::Local ? "Local" : "Remote") << std::endl;
    std::cout << "  Dispatch Kind: ";
    switch (state_->plan.kind) {
        case DispatchKind::CpuOnly:
            std::cout << "CPU Only";
            break;
        case DispatchKind::GpuOnly:
            std::cout << "GPU Only";
            break;
        case DispatchKind::CpuGpuSplit:
            std::cout << "CPU-GPU Split";
            break;
        case DispatchKind::SimOnly:
            std::cout << "SIM Only";
            break;
    }
    std::cout << std::endl;

    if (state_->plan.cpu_end > state_->plan.cpu_begin) {
        std::cout << "  CPU Work Range: [" << state_->plan.cpu_begin << ", " << state_->plan.cpu_end << ")" << std::endl;
    }
    if (state_->plan.gpu_end > state_->plan.gpu_begin) {
        std::cout << "  GPU Work Range: [" << state_->plan.gpu_begin << ", " << state_->plan.gpu_end << ")" << std::endl;
    }
    if (state_->plan.sim_end > state_->plan.sim_begin) {
        std::cout << "  SIM Work Range: [" << state_->plan.sim_begin << ", " << state_->plan.sim_end << ")" << std::endl;
    }
    std::cout << "  Estimated Cost: " << state_->plan.estimated_cost << std::endl;
}

} // namespace Routing
