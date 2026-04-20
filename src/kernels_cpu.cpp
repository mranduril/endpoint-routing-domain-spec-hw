#include "kernels.h"

#include <stdexcept>

namespace Routing
{
    void saxpy_cpu_only(float a, std::size_t n, const float* x, float* y)
    {
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = a * x[i] + y[i];
        }
    }

    void saxpy_simpy(float a, std::size_t n, const float* x, float* y)
    {
        (void)a;
        (void)n;
        (void)x;
        (void)y;
        throw std::logic_error("SimPy SAXPY endpoint is not implemented yet");
    }
}
