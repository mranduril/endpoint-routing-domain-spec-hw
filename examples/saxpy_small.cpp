#include "job.h"
#include "runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace
{
std::string run_timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_r(&now_time, &local_time);

    std::ostringstream out;
    out << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    return out.str();
}
}

int main()
{
    constexpr std::size_t n = 1024;
    constexpr float a = 2.0f;
    constexpr float initial_y = 3.0f;

    mkdir("outputs", 0775);
    mkdir("outputs/sim_traces", 0775);
    const std::string timestamp = run_timestamp();
    // Examples own their output paths. The runtime only honors the environment
    // variables and appends routing/SIM records there.
    const char* trace_path = "outputs/sim_traces/routing_saxpy_sim_jobs.jsonl";
    const std::string log_path = "outputs/routing_saxpy_run_log_" +
        timestamp + ".jsonl";
    std::ofstream(trace_path, std::ios::trunc).close();
    std::ofstream(log_path, std::ios::trunc).close();
    setenv("ROUTING_SIM_TRACE", trace_path, 1);
    setenv("ROUTING_RUN_LOG", log_path.c_str(), 1);

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

    std::fill(y->begin(), y->end(), initial_y);

    Routing::Job sim_job = Routing::make_saxpy(a, x, y);
    sim_job.metadata.job_id = 2;
    sim_job.metadata.node_id = 0;

    Routing::Request sim_request =
        Routing::submit(sim_job, Routing::RoutingPolicy::ForceSim);
    sim_request.wait();

    for (float value : *y) {
        if (std::fabs(value - expected) > 1.0e-5f) {
            std::cerr << "SAXPY SIM verification failed\n";
            return 1;
        }
    }

    std::cout << "SAXPY CPU verification passed\n";
    std::cout << "SAXPY SIM verification passed\n";
    std::cout << "SIM trace emitted to " << trace_path << '\n';
    std::cout << "Run log emitted to " << log_path << '\n';
    return 0;
}
