// This defines WorkloadType enum and related structs
#ifndef WORKLOAD_TYPE_H
#define WORKLOAD_TYPE_H

#include <cstddef>
#include <memory>
#include <vector>

namespace Routing
{
    struct payloadSAXPY {
        float a;
        std::size_t n;
        std::shared_ptr<const std::vector<float>> x;
        std::shared_ptr<std::vector<float>> y;
    };

    enum class WorkloadType {
        SAXPY,
        STENCIL,
        NOT_SUPPORTED
    };
}

#endif // workload_type.h