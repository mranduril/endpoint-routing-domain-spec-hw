// src/job.cpp
#include "job.h"

#include <variant>

namespace Routing
{
    Job make_saxpy(
        float a,
        std::shared_ptr<const std::vector<float>> x,
        std::shared_ptr<std::vector<float>> y)
    {
        payloadSAXPY saxpy;
        saxpy.a = a;
        saxpy.n = (x ? x->size() : 0);
        saxpy.x = std::move(x);
        saxpy.y = std::move(y);

        Job job;
        job.type = WorkloadType::SAXPY;
        job.payload = std::move(saxpy);
        return job;
    }

    bool Job::validate() const
    {
        switch (type) {
            case WorkloadType::SAXPY: {
                if (!std::holds_alternative<payloadSAXPY>(payload)) {
                    return false;
                }

                const auto& saxpy = std::get<payloadSAXPY>(payload);
                if (!saxpy.x || !saxpy.y) {
                    return false;
                }

                if (saxpy.n == 0) {
                    return false;
                }

                if (saxpy.x->size() != saxpy.n || saxpy.y->size() != saxpy.n) {
                    return false;
                }

                return true;
            }

            default:
                return false;
        }
    }
}
