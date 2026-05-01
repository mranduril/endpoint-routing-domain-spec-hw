#include "runtime.h"

#include "kernels.h"

#include <exception>
#include <cstdlib>
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
std::string workload_name(WorkloadType type)
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

const char* sim_trace_path()
{
    const char* configured = std::getenv("ROUTING_SIM_TRACE");
    return configured != nullptr ? configured : "traces/sim_jobs.jsonl";
}

void ensure_default_trace_dir()
{
    const char* configured = std::getenv("ROUTING_SIM_TRACE");
    if (configured == nullptr) {
        mkdir("traces", 0775);
    }
}

void emit_sim_trace_record(const Job& job)
{
    ensure_default_trace_dir();
    std::ofstream trace(sim_trace_path(), std::ios::app);
    if (!trace) {
        throw std::runtime_error("Failed to open SIM trace output");
    }

    trace << "{\"job_id\":" << job.metadata.job_id
          << ",\"node_id\":" << job.metadata.node_id
          << ",\"endpoint\":\"SIM" << job.metadata.node_id << "\""
          << ",\"op\":\"" << workload_name(job.type) << "\""
          << ",\"arrival_us\":" << job.metadata.arrival_us;

    if (std::holds_alternative<payloadSAXPY>(job.payload)) {
        const auto& saxpy = std::get<payloadSAXPY>(job.payload);
        trace << ",\"n\":" << saxpy.n
              << ",\"input_bytes\":" << saxpy.n * sizeof(float) * 2
              << ",\"output_bytes\":" << saxpy.n * sizeof(float)
              << ",\"metadata\":{\"dtype\":\"float32\",\"layout\":\"contiguous\"}";
    } else if (std::holds_alternative<payloadJacobi>(job.payload)) {
        const auto& jacobi = std::get<payloadJacobi>(job.payload);
        const std::size_t grid_bytes = jacobi.nx * jacobi.ny * sizeof(float);
        trace << ",\"nx\":" << jacobi.nx
              << ",\"ny\":" << jacobi.ny
              << ",\"halo_width\":" << jacobi.halo_width
              << ",\"input_bytes\":" << grid_bytes
              << ",\"output_bytes\":" << grid_bytes
              << ",\"metadata\":{\"stencil\":\"2d_5pt\",\"dtype\":\"float32\","
              << "\"layout\":\"row_major\",\"iteration\":" << job.metadata.iteration
              << ",\"neighbor_node_id\":" << job.metadata.neighbor_node_id << "}";
    }

    trace << "}\n";
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
                            emit_sim_trace_record(state->job);
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
                            emit_sim_trace_record(state->job);
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
