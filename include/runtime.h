// Signatures of runtime APIs.
#ifndef RUNTIME_H
#define RUNTIME_H

#include "job.h"
#include "router.h"

namespace Routing
{
    class Request
    {
        // This is a placeholder for the return type of submit. It can be used to query the status of the job, or to retrieve results when the job is done.
    };
    // This should contain APIs that users can use. For example, defining a Job and submitting it
    Request submit(const Job& job, const RoutingPolicy& endpoint);
    void wait(const Request& request);
    bool ready(const Request& request);
}

#endif // runtime.h