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
        JobMetadata metadata;
        std::variant<payloadSAXPY, payloadJacobi> payload;

        bool validate() const;
    };
    
    Job make_saxpy(
        float a,
        std::shared_ptr<const std::vector<float>> x,
        std::shared_ptr<std::vector<float>> y
    );

    Job make_jacobi(
        WorkloadType type,
        std::size_t nx,
        std::size_t ny,
        std::size_t halo_width,
        std::shared_ptr<const std::vector<float>> input,
        std::shared_ptr<std::vector<float>> output,
        JobMetadata metadata = {}
    );
}

#endif // job.h
