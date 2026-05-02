#include "runtime.h"

#include "kernels.h"

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <sys/stat.h>

namespace Routing
{
namespace
{
const char* workload_name(WorkloadType type)
{
    switch (type) {
        case WorkloadType::SAXPY:
            return "SAXPY";
        case WorkloadType::JACOBI_INTERIOR:
            return "JACOBI_INTERIOR";
        case WorkloadType::JACOBI_BOUNDARY:
            return "JACOBI_BOUNDARY";
        case WorkloadType::JACOBI_HALO_BOUNDARY:
            return "JACOBI_HALO_BOUNDARY";
        default:
            return "NOT_SUPPORTED";
    }
}

const char* dispatch_name(DispatchKind kind)
{
    switch (kind) {
        case DispatchKind::CpuOnly:
            return "CpuOnly";
        case DispatchKind::GpuOnly:
            return "GpuOnly";
        case DispatchKind::CpuGpuSplit:
            return "CpuGpuSplit";
        case DispatchKind::SimOnly:
            return "SimOnly";
        default:
            return "Unknown";
    }
}

const char* node_name(NodeKind kind)
{
    switch (kind) {
        case NodeKind::Local:
            return "Local";
        case NodeKind::Remote:
            return "Remote";
        default:
            return "Unknown";
    }
}

const char* run_log_path()
{
    const char* configured = std::getenv("ROUTING_RUN_LOG");
    return configured != nullptr ? configured : "outputs/routing_run_log.jsonl";
}

void ensure_output_dir()
{
    mkdir("outputs", 0775);
}

void write_plan_ranges_json(std::ostream& out, const DispatchPlan& plan)
{
    out << "\"ranges\":{"
        << "\"cpu\":[" << plan.cpu_begin << "," << plan.cpu_end << "]"
        << ",\"gpu\":[" << plan.gpu_begin << "," << plan.gpu_end << "]"
        << ",\"sim\":[" << plan.sim_begin << "," << plan.sim_end << "]"
        << "}";
}

void write_cost_model_json(std::ostream& out, const DispatchPlan& plan)
{
    out << "\"cost_model\":{"
        << "\"cpu\":" << plan.cpu_estimated_cost
        << ",\"gpu\":" << plan.gpu_estimated_cost
        << ",\"split\":" << plan.split_estimated_cost
        << ",\"sim\":" << plan.sim_estimated_cost
        << "}";
}

void write_dispatch_plan_json_fields(std::ostream& out, const DispatchPlan& plan)
{
    out << "\"node_kind\":\"" << node_name(plan.node_kind) << "\""
        << ",\"decision\":\"" << dispatch_name(plan.kind) << "\""
        << ",\"estimated_cost\":" << plan.estimated_cost
        << ",\"work_units\":" << plan.work_units
        << ",";
    write_cost_model_json(out, plan);
    out << ",";
    write_plan_ranges_json(out, plan);
}

void print_work_range(
    std::ostream& out,
    const char* label,
    std::size_t begin,
    std::size_t end)
{
    if (end > begin) {
        out << "  " << label << " Work Range: ["
            << begin << ", " << end << ")\n";
    }
}

void print_dispatch_plan(std::ostream& out, const DispatchPlan& plan)
{
    out << "Dispatch Plan:\n"
        << "  Node Kind: " << node_name(plan.node_kind) << '\n'
        << "  Dispatch Kind: " << dispatch_name(plan.kind) << '\n';

    print_work_range(out, "CPU", plan.cpu_begin, plan.cpu_end);
    print_work_range(out, "GPU", plan.gpu_begin, plan.gpu_end);
    print_work_range(out, "SIM", plan.sim_begin, plan.sim_end);

    out << "  Estimated Cost: " << plan.estimated_cost << '\n'
        << "  Work Units: " << plan.work_units << '\n'
        << "  Cost Model: "
        << "cpu=" << plan.cpu_estimated_cost
        << ", gpu=" << plan.gpu_estimated_cost
        << ", split=" << plan.split_estimated_cost
        << ", sim=" << plan.sim_estimated_cost << '\n';
}

void write_routing_log(
    const Job& job,
    RoutingPolicy policy,
    const DispatchPlan& plan)
{
    ensure_output_dir();
    std::ofstream log(run_log_path(), std::ios::app);
    if (!log) {
        throw std::runtime_error("Failed to open routing run log");
    }

    log << "{\"job_id\":" << job.metadata.job_id
        << ",\"node_id\":" << job.metadata.node_id
        << ",\"iteration\":" << job.metadata.iteration
        << ",\"neighbor_node_id\":" << job.metadata.neighbor_node_id
        << ",\"op\":\"" << workload_name(job.type) << "\""
        << ",\"policy\":\"" << RoutingPolicyNames.at(policy) << "\""
        << ",";
    write_dispatch_plan_json_fields(log, plan);
    log << "}\n";
}

void run_jacobi_cpu(WorkloadType type, const payloadJacobi& jacobi)
{
    if (type == WorkloadType::JACOBI_INTERIOR) {
        jacobi_interior_cpu(
            jacobi.nx,
            jacobi.ny,
            jacobi.halo_width,
            jacobi.input->data(),
            jacobi.output->data());
        return;
    }

    jacobi_boundary_cpu(
        jacobi.nx,
        jacobi.ny,
        jacobi.halo_width,
        jacobi.input->data(),
        jacobi.output->data());
}

void run_jacobi_simpy(
    WorkloadType type,
    const payloadJacobi& jacobi,
    const JobMetadata& metadata)
{
    if (type == WorkloadType::JACOBI_INTERIOR) {
        jacobi_interior_simpy(
            jacobi.nx,
            jacobi.ny,
            jacobi.halo_width,
            jacobi.input->data(),
            jacobi.output->data(),
            metadata);
        return;
    }

    if (type == WorkloadType::JACOBI_BOUNDARY) {
        jacobi_boundary_simpy(
            jacobi.nx,
            jacobi.ny,
            jacobi.halo_width,
            jacobi.input->data(),
            jacobi.output->data(),
            metadata);
        return;
    }

    jacobi_halo_boundary_simpy(
        jacobi.nx,
        jacobi.ny,
        jacobi.halo_width,
        jacobi.input->data(),
        jacobi.output->data(),
        metadata);
}
}

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
    write_routing_log(state->job, policy, state->plan);
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
                        case DispatchKind::GpuOnly:
                            saxpy_gpu_only(
                                saxpy.a,
                                saxpy.n,
                                saxpy.x->data(),
                                saxpy.y->data());
                            break;
                        case DispatchKind::CpuGpuSplit:
                        {
                            const std::size_t n = saxpy.n;
                            const std::size_t cpu_begin = state->plan.cpu_begin;
                            const std::size_t cpu_end = state->plan.cpu_end;
                            const std::size_t gpu_begin = state->plan.gpu_begin;
                            const std::size_t gpu_end = state->plan.gpu_end;

                            if (cpu_begin > cpu_end || gpu_begin > gpu_end ||
                                cpu_end > n || gpu_end > n) {
                                throw std::invalid_argument("CpuGpuSplit plan has invalid slice bounds");
                            }

                            const std::size_t cpu_count = cpu_end - cpu_begin;
                            const std::size_t gpu_count = gpu_end - gpu_begin;
                            std::exception_ptr cpu_error;

                            std::thread cpu_thread([&]() {
                                try {
                                    if (cpu_count > 0) {
                                        saxpy_cpu_only(
                                            saxpy.a,
                                            cpu_count,
                                            saxpy.x->data() + cpu_begin,
                                            saxpy.y->data() + cpu_begin);
                                    }
                                } catch (...) {
                                    cpu_error = std::current_exception();
                                }
                            });

                            try {
                                if (gpu_count > 0) {
                                    saxpy_gpu_only(
                                        saxpy.a,
                                        gpu_count,
                                        saxpy.x->data() + gpu_begin,
                                        saxpy.y->data() + gpu_begin);
                                }
                            } catch (...) {
                                cpu_thread.join();
                                if (cpu_error) {
                                    std::rethrow_exception(cpu_error);
                                }
                                throw;
                            }

                            cpu_thread.join();
                            if (cpu_error) {
                                std::rethrow_exception(cpu_error);
                            }
                            break;
                        }
                        case DispatchKind::SimOnly:
                            saxpy_simpy(
                                saxpy.a,
                                saxpy.n,
                                saxpy.x->data(),
                                saxpy.y->data(),
                                state->job.metadata);
                            break;
                        default:
                            throw std::logic_error("Selected dispatch plan is not implemented yet");
                    }
                    break;
                }
                case WorkloadType::JACOBI_INTERIOR:
                case WorkloadType::JACOBI_BOUNDARY:
                case WorkloadType::JACOBI_HALO_BOUNDARY: {
                    if (!std::holds_alternative<payloadJacobi>(state->job.payload)) {
                        throw std::invalid_argument("Job payload does not match Jacobi workload");
                    }

                    auto& jacobi = std::get<payloadJacobi>(state->job.payload);
                    switch (state->plan.kind) {
                        case DispatchKind::CpuOnly:
                            run_jacobi_cpu(state->job.type, jacobi);
                            break;
                        case DispatchKind::GpuOnly:
                            if (state->job.type != WorkloadType::JACOBI_INTERIOR) {
                                throw std::logic_error("GPU Jacobi boundary path is not implemented yet");
                            }
                            jacobi_interior_gpu(
                                jacobi.nx,
                                jacobi.ny,
                                jacobi.halo_width,
                                jacobi.input->data(),
                                jacobi.output->data());
                            break;
                        case DispatchKind::SimOnly:
                            run_jacobi_simpy(
                                state->job.type,
                                jacobi,
                                state->job.metadata);
                            break;
                        default:
                            throw std::logic_error("Selected Jacobi dispatch plan is not implemented yet");
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
    if (!state_) {
        throw std::invalid_argument("Cannot inspect an invalid request");
    }

    print_dispatch_plan(std::cout, state_->plan);
}

} // namespace Routing
