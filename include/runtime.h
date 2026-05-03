// Runtime API. submit() routes a logical Job and launches endpoint execution in
// a detached worker thread; Request is the handle used to wait/query completion.
#ifndef RUNTIME_H
#define RUNTIME_H

#pragma once

#include "job.h"
#include "router.h"
#include <memory>
#include <mutex>
#include <condition_variable>
#include <vector>

namespace Routing
{
    enum class RequestStatus {
        Pending,
        Running,
        Completed,
        Failed
    };

    class DistributedRuntime
    {
        public:
            DistributedRuntime(int* argc = nullptr, char*** argv = nullptr);
            ~DistributedRuntime();

            DistributedRuntime(const DistributedRuntime&) = delete;
            DistributedRuntime& operator=(const DistributedRuntime&) = delete;

            int rank() const { return rank_; }
            int size() const { return size_; }
            int node_id() const { return rank_; }
            bool distributed() const { return size_ > 1; }

            void exchange_jacobi_halos(
                std::size_t nx,
                std::size_t owned_rows,
                std::vector<float>& local_tile,
                int neighbor_node_id) const;

            std::vector<float> gather_jacobi_global(
                const std::vector<float>& local_tile,
                std::size_t nx,
                std::size_t owned_rows,
                std::size_t start_global_row,
                std::size_t global_ny,
                const std::vector<float>& boundary_source) const;

        private:
            int rank_ = 0;
            int size_ = 1;
            bool owns_mpi_ = false;
    };

    class Request
    {
        public:
            Request() = default;

            bool valid() const { return static_cast<bool>(state_); }

            // Users can call wait() to block until the job is done
            void wait();

            // Users can call ready() to check if the job is done without blocking
            bool ready();

            void inspect_dispatch_plan();

        private:
            struct State {
                // State is shared with the detached worker thread. Request is
                // copyable via shared_ptr ownership and wait() observes status.
                Job job;
                DispatchPlan plan{};
                RequestStatus status = RequestStatus::Pending;
                std::exception_ptr error;
                mutable std::mutex mutex;
                std::condition_variable cv;
            };
        
            explicit Request(std::shared_ptr<State> state)
                : state_(std::move(state)) {}
        
            std::shared_ptr<State> state_;

            friend Request submit(Job job, RoutingPolicy policy);
            friend Request submit(Job job,
                                  RoutingPolicy policy,
                                  RouterConfig config);
    };
    // This should contain APIs that users can use. For example, defining a Job and submitting it
    Request submit(Job job, RoutingPolicy policy = RoutingPolicy::Auto);
    Request submit(Job job,
                   RoutingPolicy policy,
                   RouterConfig config);

    void mark_data_location(const void* buffer, DataLocation location);
    DataLocation lookup_data_location(const void* buffer);
}

#endif // runtime.h
