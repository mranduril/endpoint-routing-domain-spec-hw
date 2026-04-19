// Header definition of a Job for the user to implement
#ifndef JOB_H
#define JOB_H

#include "workload_type.h"
#include <vector>
#include <variant>

namespace Routing
{
    struct Job
    {
        WorkloadType type;  // Carries the type of workload
        std::variant<payloadSAXPY> payload; // Add more payload types as needed

        bool validate();
    };
}
#endif // job.h