#include "job.h"
#include "runtime.h"

#include <chrono>
#include <iostream>
#include <random>
#include <unordered_set>

int main() {
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;

    const auto total_start = Clock::now();

    float a = 2.0f;
    float initial_y = 2.0f;

    const auto buffer_start = Clock::now();
    std::shared_ptr<std::vector<float>> x = std::make_shared<std::vector<float>>(100000000, 1.0f);
    std::shared_ptr<std::vector<float>> y = std::make_shared<std::vector<float>>(100000000, initial_y);
    const auto buffer_end = Clock::now();

    const auto job_start = Clock::now();
    Routing::Job cpu_saxpy = Routing::make_saxpy(
        a,
        x,
        y
    );
    const auto job_end = Clock::now();

    const auto submit_start = Clock::now();
    Routing::Request request = Routing::submit(cpu_saxpy, Routing::RoutingPolicy::ForceCpu);
    const auto submit_end = Clock::now();

    const auto wait_start = Clock::now();
    request.wait();
    const auto wait_end = Clock::now();

    if (request.ready()) {
        const auto verify_start = Clock::now();
        const std::size_t sample_count = std::min<std::size_t>(100, x->size());
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<std::size_t> dist(0, x->size() - 1);
        std::unordered_set<std::size_t> sampled_indices;
        sampled_indices.reserve(sample_count);

        while (sampled_indices.size() < sample_count) {
            sampled_indices.insert(dist(rng));
        }

        // Check a random sample of results instead of scanning the full output.
        for (std::size_t i : sampled_indices) {
            if ((*y)[i] != a * (*x)[i] + initial_y) {
                std::cerr << "Error at index " << i << ": expected " << a * (*x)[i] + initial_y
                          << ", got " << (*y)[i] << std::endl;
                return 1;
            }
        }
        const auto verify_end = Clock::now();
        const auto total_end = Clock::now();

        std::cout << "Vector length: " << x->size() << '\n';
        std::cout << "Buffer allocation/init: "
                  << Milliseconds(buffer_end - buffer_start).count() << " ms\n";
        std::cout << "Job creation: "
                  << Milliseconds(job_end - job_start).count() << " ms\n";
        std::cout << "Submit latency: "
                  << Milliseconds(submit_end - submit_start).count() << " ms\n";
        std::cout << "Wait/execution: "
                  << Milliseconds(wait_end - wait_start).count() << " ms\n";
        std::cout << "Verification: "
                  << Milliseconds(verify_end - verify_start).count() << " ms\n";
        std::cout << "Verification samples: " << sample_count << '\n';
        std::cout << "Total runtime: "
                  << Milliseconds(total_end - total_start).count() << " ms\n";
        std::cout << "SAXPY computation completed successfully!" << std::endl;
        request.inspect_dispatch_plan();
    } else {
        std::cerr << "SAXPY computation failed." << std::endl;
        return 1;
    }
    return 0;
}
