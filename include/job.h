// Public logical job API. Users construct Jobs here; the router maps them to
// physical endpoint implementations in CPU/GPU/SIM kernels.
#ifndef JOB_H
#define JOB_H

#include "workload_type.h"
#include <vector>
#include <variant>

namespace Routing
{
    struct Job
    {
        WorkloadType type;
        // Metadata is deliberately separate from the payload so scheduling
        // hints can evolve without changing the math payload layout.
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
