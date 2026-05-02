// This defines WorkloadType enum and related structs
#ifndef WORKLOAD_TYPE_H
#define WORKLOAD_TYPE_H

#include <cstddef>
#include <memory>
#include <vector>

namespace Routing
{
    struct JobMetadata {
        std::size_t job_id = 0;
        int node_id = 0;
        int iteration = 0;
        int neighbor_node_id = -1;
    };

    struct payloadSAXPY {
        float a;
        std::size_t n;
        std::shared_ptr<const std::vector<float>> x;
        std::shared_ptr<std::vector<float>> y;
    };

    struct payloadJacobi {
        std::size_t nx;
        std::size_t ny;
        std::size_t halo_width;
        std::shared_ptr<const std::vector<float>> input;
        std::shared_ptr<std::vector<float>> output;
    };

    enum class WorkloadType {
        SAXPY,
        JACOBI_INTERIOR,
        JACOBI_BOUNDARY,
        JACOBI_HALO_BOUNDARY,
        NOT_SUPPORTED
    };
}

#endif // workload_type.h
