#include "runtime.h"

#include "kernels.h"

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <sys/stat.h>

namespace Routing
{
namespace
{
struct EndpointPressure {
    std::size_t cpu_jobs = 0;
    std::size_t gpu_jobs = 0;
    std::size_t sim_jobs = 0;
};

std::mutex g_pressure_mutex;
std::unordered_map<int, EndpointPressure> g_endpoint_pressure;
std::mutex g_data_location_mutex;
std::unordered_map<const void*, DataLocation> g_data_locations;

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

const char* data_location_name(DataLocation location)
{
    switch (location) {
        case DataLocation::Host:
            return "Host";
        case DataLocation::GPU:
            return "GPU";
        case DataLocation::SIM:
            return "SIM";
        case DataLocation::Remote:
            return "Remote";
        case DataLocation::Unknown:
            return "Unknown";
        default:
            return "Unknown";
    }
}

const char* run_log_path()
{
    const char* configured = std::getenv("ROUTING_RUN_LOG");
    return configured != nullptr ? configured : "outputs/routing_run_log.jsonl";
}

const void* input_buffer_id(const Job& job)
{
    if (job.type == WorkloadType::SAXPY &&
        std::holds_alternative<payloadSAXPY>(job.payload)) {
        const auto& saxpy = std::get<payloadSAXPY>(job.payload);
        return saxpy.x ? static_cast<const void*>(saxpy.x->data()) : nullptr;
    }

    if ((job.type == WorkloadType::JACOBI_INTERIOR ||
         job.type == WorkloadType::JACOBI_BOUNDARY ||
         job.type == WorkloadType::JACOBI_HALO_BOUNDARY) &&
        std::holds_alternative<payloadJacobi>(job.payload)) {
        const auto& jacobi = std::get<payloadJacobi>(job.payload);
        return jacobi.input ? static_cast<const void*>(jacobi.input->data()) : nullptr;
    }

    return nullptr;
}

const void* output_buffer_id(const Job& job)
{
    if (job.type == WorkloadType::SAXPY &&
        std::holds_alternative<payloadSAXPY>(job.payload)) {
        const auto& saxpy = std::get<payloadSAXPY>(job.payload);
        return saxpy.y ? static_cast<const void*>(saxpy.y->data()) : nullptr;
    }

    if ((job.type == WorkloadType::JACOBI_INTERIOR ||
         job.type == WorkloadType::JACOBI_BOUNDARY ||
         job.type == WorkloadType::JACOBI_HALO_BOUNDARY) &&
        std::holds_alternative<payloadJacobi>(job.payload)) {
        const auto& jacobi = std::get<payloadJacobi>(job.payload);
        return jacobi.output ? static_cast<const void*>(jacobi.output->data()) : nullptr;
    }

    return nullptr;
}

bool tracked_location(const void* buffer, DataLocation& location)
{
    if (buffer == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_data_location_mutex);
    const auto it = g_data_locations.find(buffer);
    if (it == g_data_locations.end()) {
        return false;
    }

    location = it->second;
    return true;
}

void track_location_if_missing(const void* buffer, DataLocation location)
{
    if (buffer == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_data_location_mutex);
    g_data_locations.emplace(buffer, location);
}

void apply_data_residency(Job& job)
{
    track_location_if_missing(input_buffer_id(job), job.metadata.input_location);
    track_location_if_missing(output_buffer_id(job), job.metadata.output_location);

    DataLocation location = DataLocation::Unknown;
    if (tracked_location(input_buffer_id(job), location)) {
        job.metadata.input_location = location;
    }
    if (tracked_location(output_buffer_id(job), location)) {
        job.metadata.output_location = location;
    }
}

void record_output_residency(const Job& job)
{
    // Current CPU/GPU/SIM wrappers all leave user-visible vectors on host:
    // GPU copies results back and SIM uses CPU fallback for correctness.
    mark_data_location(output_buffer_id(job), DataLocation::Host);
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

bool is_unavailable_cost(double cost)
{
    return cost >= std::numeric_limits<double>::max() / 8.0;
}

void write_cost_value_json(std::ostream& out, double cost)
{
    if (is_unavailable_cost(cost)) {
        out << "null";
        return;
    }
    out << cost;
}

void write_cost_model_json(std::ostream& out, const DispatchPlan& plan)
{
    out << "\"cost_model\":{";
    out << "\"cpu\":";
    write_cost_value_json(out, plan.cpu_estimated_cost);
    out << ",\"gpu\":";
    write_cost_value_json(out, plan.gpu_estimated_cost);
    out << ",\"split\":";
    write_cost_value_json(out, plan.split_estimated_cost);
    out << ",\"sim\":";
    write_cost_value_json(out, plan.sim_estimated_cost);
    out << ",\"remote\":";
    write_cost_value_json(out, plan.remote_estimated_cost);
    out << "}";
}

void write_dispatch_plan_json_fields(std::ostream& out, const DispatchPlan& plan)
{
    out << "\"node_kind\":\"" << node_name(plan.node_kind) << "\""
        << ",\"target_node_id\":" << plan.target_node_id
        << ",\"decision\":\"" << dispatch_name(plan.kind) << "\""
        << ",\"estimated_cost\":" << plan.estimated_cost
        << ",\"work_units\":" << plan.work_units
        << ",\"input_bytes\":" << plan.input_bytes
        << ",\"output_bytes\":" << plan.output_bytes
        << ",";
    write_cost_model_json(out, plan);
    out << ",";
    write_plan_ranges_json(out, plan);
}

RouterConfig runtime_router_config(
    RouterConfig base_config,
    int target_node_id)
{
    // Snapshot queue pressure at planning time. Pressure is keyed by logical
    // target node so a two-logical-node run can model CPU0/SIM0 separately from
    // CPU1/SIM1 even when both live inside one physical process.
    RouterConfig config = std::move(base_config);
    std::lock_guard<std::mutex> lock(g_pressure_mutex);
    const auto it = g_endpoint_pressure.find(target_node_id);
    if (it != g_endpoint_pressure.end()) {
        config.queued_cpu_jobs += it->second.cpu_jobs;
        config.queued_gpu_jobs += it->second.gpu_jobs;
        config.queued_sim_jobs += it->second.sim_jobs;
    }
    return config;
}

void increment_pressure(const DispatchPlan& plan)
{
    // These counters are an intentionally lightweight approximation of queue
    // pressure. The next planning decision sees currently running work and can
    // bias away from a busy endpoint.
    std::lock_guard<std::mutex> lock(g_pressure_mutex);
    EndpointPressure& pressure = g_endpoint_pressure[plan.target_node_id];
    switch (plan.kind) {
        case DispatchKind::CpuOnly:
            ++pressure.cpu_jobs;
            break;
        case DispatchKind::GpuOnly:
            ++pressure.gpu_jobs;
            break;
        case DispatchKind::CpuGpuSplit:
            ++pressure.cpu_jobs;
            ++pressure.gpu_jobs;
            break;
        case DispatchKind::SimOnly:
            ++pressure.sim_jobs;
            break;
    }
}

void decrement_pressure(const DispatchPlan& plan)
{
    std::lock_guard<std::mutex> lock(g_pressure_mutex);
    EndpointPressure& pressure = g_endpoint_pressure[plan.target_node_id];
    switch (plan.kind) {
        case DispatchKind::CpuOnly:
            --pressure.cpu_jobs;
            break;
        case DispatchKind::GpuOnly:
            --pressure.gpu_jobs;
            break;
        case DispatchKind::CpuGpuSplit:
            --pressure.cpu_jobs;
            --pressure.gpu_jobs;
            break;
        case DispatchKind::SimOnly:
            --pressure.sim_jobs;
            break;
    }
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
        << "  Target Node ID: " << plan.target_node_id << '\n'
        << "  Work Units: " << plan.work_units << '\n'
        << "  Input Bytes: " << plan.input_bytes << '\n'
        << "  Output Bytes: " << plan.output_bytes << '\n'
        << "  Cost Model: "
        << "cpu=" << plan.cpu_estimated_cost
        << ", gpu=" << plan.gpu_estimated_cost
        << ", split=" << plan.split_estimated_cost
        << ", sim=" << plan.sim_estimated_cost
        << ", remote=" << plan.remote_estimated_cost << '\n';
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
        << ",\"data_location\":{\"input\":\""
        << data_location_name(job.metadata.input_location)
        << "\",\"output\":\""
        << data_location_name(job.metadata.output_location)
        << "\"}"
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

DistributedRuntime::DistributedRuntime(int* argc, char*** argv)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = 0;
        int local_argc = 0;
        char** local_argv = nullptr;
        int* init_argc = argc != nullptr ? argc : &local_argc;
        char*** init_argv = argv != nullptr ? argv : &local_argv;
        MPI_Init_thread(init_argc, init_argv, MPI_THREAD_SERIALIZED, &provided);
        owns_mpi_ = true;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

DistributedRuntime::~DistributedRuntime()
{
    if (!owns_mpi_) {
        return;
    }

    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

void DistributedRuntime::exchange_jacobi_halos(
    std::size_t nx,
    std::size_t owned_rows,
    std::vector<float>& local_tile,
    int neighbor_node_id) const
{
    if (local_tile.size() < nx * (owned_rows + 2)) {
        throw std::invalid_argument("Jacobi local tile is too small for halo exchange");
    }

    jacobi_exchange_halos_cpu(
        nx,
        owned_rows,
        local_tile.data(),
        node_id(),
        neighbor_node_id);
    mark_data_location(local_tile.data(), DataLocation::Host);
}

std::vector<float> DistributedRuntime::gather_jacobi_global(
    const std::vector<float>& local_tile,
    std::size_t nx,
    std::size_t owned_rows,
    std::size_t start_global_row,
    std::size_t global_ny,
    const std::vector<float>& boundary_source) const
{
    if (local_tile.size() < nx * (owned_rows + 2)) {
        throw std::invalid_argument("Jacobi local tile is too small to gather");
    }

    if (boundary_source.size() != nx * global_ny) {
        throw std::invalid_argument("Jacobi boundary source size does not match global grid");
    }

    if (size_ == 1) {
        std::vector<float> global = boundary_source;
        for (std::size_t row = 0; row < owned_rows; ++row) {
            std::copy_n(
                local_tile.data() + (row + 1) * nx,
                nx,
                global.data() + (start_global_row + row) * nx);
        }
        return global;
    }

    const unsigned long long local_rows =
        static_cast<unsigned long long>(owned_rows);
    const unsigned long long local_start =
        static_cast<unsigned long long>(start_global_row);
    std::vector<unsigned long long> rows_by_rank(size_);
    std::vector<unsigned long long> starts_by_rank(size_);

    MPI_Gather(
        &local_rows,
        1,
        MPI_UNSIGNED_LONG_LONG,
        rows_by_rank.data(),
        1,
        MPI_UNSIGNED_LONG_LONG,
        0,
        MPI_COMM_WORLD);
    MPI_Gather(
        &local_start,
        1,
        MPI_UNSIGNED_LONG_LONG,
        starts_by_rank.data(),
        1,
        MPI_UNSIGNED_LONG_LONG,
        0,
        MPI_COMM_WORLD);

    const int send_count = static_cast<int>(owned_rows * nx);
    std::vector<int> recv_counts;
    std::vector<int> displacements;
    std::vector<float> gathered;
    if (rank_ == 0) {
        recv_counts.resize(size_);
        displacements.resize(size_);
        int offset = 0;
        for (int rank = 0; rank < size_; ++rank) {
            recv_counts[rank] =
                static_cast<int>(rows_by_rank[rank] * nx);
            displacements[rank] = offset;
            offset += recv_counts[rank];
        }
        gathered.resize(static_cast<std::size_t>(offset));
    }

    MPI_Gatherv(
        local_tile.data() + nx,
        send_count,
        MPI_FLOAT,
        rank_ == 0 ? gathered.data() : nullptr,
        rank_ == 0 ? recv_counts.data() : nullptr,
        rank_ == 0 ? displacements.data() : nullptr,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    if (rank_ != 0) {
        return {};
    }

    std::vector<float> global = boundary_source;
    for (int rank = 0; rank < size_; ++rank) {
        const std::size_t rows =
            static_cast<std::size_t>(rows_by_rank[rank]);
        const std::size_t start =
            static_cast<std::size_t>(starts_by_rank[rank]);
        const float* source = gathered.data() + displacements[rank];
        for (std::size_t row = 0; row < rows; ++row) {
            std::copy_n(
                source + row * nx,
                nx,
                global.data() + (start + row) * nx);
        }
    }
    return global;
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
    return submit(std::move(job), policy, RouterConfig{});
}

Request submit(Job job, RoutingPolicy policy, RouterConfig config)
{
    if (!job.validate()) {
        throw std::invalid_argument("Invalid job");
    }

    apply_data_residency(job);

    auto state = std::make_shared<Request::State>();
    state->job = std::move(job);

    // Planning happens synchronously so callers can inspect/log the selected
    // dispatch immediately, while actual endpoint work runs asynchronously.
    Router router(runtime_router_config(std::move(config), state->job.metadata.node_id));
    state->plan = router.plan(state->job, policy);
    write_routing_log(state->job, policy, state->plan);
    increment_pressure(state->plan);
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

            record_output_residency(state->job);
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

        // Keep queue-pressure counters balanced for both success and failure.
        decrement_pressure(state->plan);
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

void mark_data_location(const void* buffer, DataLocation location)
{
    if (buffer == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_data_location_mutex);
    g_data_locations[buffer] = location;
}

DataLocation lookup_data_location(const void* buffer)
{
    if (buffer == nullptr) {
        return DataLocation::Unknown;
    }

    std::lock_guard<std::mutex> lock(g_data_location_mutex);
    const auto it = g_data_locations.find(buffer);
    return it == g_data_locations.end() ?
        DataLocation::Unknown : it->second;
}

} // namespace Routing
