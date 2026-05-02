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

namespace Routing
{
    enum class RequestStatus {
        Pending,
        Running,
        Completed,
        Failed
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
    };
    // This should contain APIs that users can use. For example, defining a Job and submitting it
    Request submit(Job job, RoutingPolicy policy = RoutingPolicy::Auto);
}

#endif // runtime.h
