#include "job.h"
#include "runtime.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

int main()
{
    constexpr std::size_t n = 1024;
    constexpr float a = 2.0f;
    constexpr float initial_y = 3.0f;

    auto x = std::make_shared<std::vector<float>>(n, 4.0f);
    auto y = std::make_shared<std::vector<float>>(n, initial_y);

    Routing::Job cpu_job = Routing::make_saxpy(a, x, y);
    cpu_job.metadata.job_id = 1;
    cpu_job.metadata.node_id = 0;

    Routing::Request cpu_request =
        Routing::submit(cpu_job, Routing::RoutingPolicy::ForceCpu);
    cpu_request.wait();

    const float expected = a * 4.0f + initial_y;
    for (float value : *y) {
        if (std::fabs(value - expected) > 1.0e-5f) {
            std::cerr << "SAXPY CPU verification failed\n";
            return 1;
        }
    }

    const char* trace_path = "/tmp/routing_saxpy_sim_jobs.jsonl";
    std::ofstream(trace_path, std::ios::trunc).close();
    setenv("ROUTING_SIM_TRACE", trace_path, 1);

    Routing::Job sim_job = Routing::make_saxpy(a, x, y);
    sim_job.metadata.job_id = 2;
    sim_job.metadata.node_id = 0;
    sim_job.metadata.arrival_us = 25.0;

    Routing::Request sim_request =
        Routing::submit(sim_job, Routing::RoutingPolicy::ForceSim);
    sim_request.wait();

    std::cout << "SAXPY CPU verification passed\n";
    std::cout << "SIM trace emitted to " << trace_path << '\n';
    return 0;
}
