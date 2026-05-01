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

    Job make_jacobi(
        WorkloadType type,
        std::size_t nx,
        std::size_t ny,
        std::size_t halo_width,
        std::shared_ptr<const std::vector<float>> input,
        std::shared_ptr<std::vector<float>> output,
        JobMetadata metadata)
    {
        payloadJacobi jacobi;
        jacobi.nx = nx;
        jacobi.ny = ny;
        jacobi.halo_width = halo_width;
        jacobi.input = std::move(input);
        jacobi.output = std::move(output);

        Job job;
        job.type = type;
        job.metadata = metadata;
        job.payload = std::move(jacobi);
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
            case WorkloadType::JACOBI_INTERIOR:
            case WorkloadType::JACOBI_BOUNDARY:
            case WorkloadType::JACOBI_HALO_BOUNDARY: {
                if (!std::holds_alternative<payloadJacobi>(payload)) {
                    return false;
                }

                const auto& jacobi = std::get<payloadJacobi>(payload);
                if (!jacobi.input || !jacobi.output) {
                    return false;
                }

                if (jacobi.nx == 0 || jacobi.ny == 0 || jacobi.halo_width == 0) {
                    return false;
                }

                const std::size_t total_cells = jacobi.nx * jacobi.ny;
                if (jacobi.input->size() != total_cells ||
                    jacobi.output->size() != total_cells) {
                    return false;
                }

                return true;
            }

            default:
                return false;
        }
    }
}
