#pragma once

#include "algorithm.hpp"
#include "cl_common.hpp"

#include <string>
#include <vector>

struct benchmark_config
{
    size_t n = 1 << 24;          // problem size (meaning is up to the algorithm)
    int warmup = 3;              // untimed runs (first runs pay one-time driver costs)
    int iters = 20;              // timed runs
    std::string kernel_dir;      // where .cl files live
    std::string extra_options;   // appended to the algorithm's build options
};

struct benchmark_result
{
    std::string name;
    size_t n = 0;
    double build_ms = 0;         // clBuildProgram time (runtime compile from source)
    double setup_ms = 0;         // host wall time for alloc + upload + setArg
    std::vector<double> kernel_ms; // per-iteration GPU time from event profiling
    double min_ms = 0, median_ms = 0, mean_ms = 0;
    double work_units = 0;
    std::string throughput_unit;
    bool verified = false;
};

class algorithm_driver
{
public:
    // platform_index / device_index < 0 => auto-pick (first GPU found, else first device).
    explicit algorithm_driver(int platform_index = -1, int device_index = -1);

    // Reads a .cl file and compiles it for this driver's device at runtime.
    cl::Program load_program(const std::string& path, const std::string& options,
                             double* build_ms = nullptr);

    benchmark_result benchmark(Algorithm& algo, const benchmark_config& cfg);

    const cl::Device& device() const { return device_; }
    const cl::Platform& platform() const { return platform_; }

    static void list_devices();

private:
    cl::Platform platform_;
    cl::Device device_;
    cl::Context context_;
    cl::CommandQueue queue_;     // created with CL_QUEUE_PROFILING_ENABLE
};

void print_report(const benchmark_result& r, const algorithm_driver& driver);
