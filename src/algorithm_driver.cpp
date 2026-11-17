#include "algorithm_driver.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::vector<cl::Device> all_devices(const cl::Platform& platform)
{
    std::vector<cl::Device> devices;
    // CL_DEVICE_TYPE_ALL never fails with CL_DEVICE_NOT_FOUND on a valid platform,
    // unlike asking for a specific type (which would throw with exceptions enabled).
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    return devices;
}

} // namespace

algorithm_driver::algorithm_driver(int platform_index, int device_index)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
        throw std::runtime_error("No OpenCL platforms found");

    if (platform_index >= 0) {
        if (platform_index >= static_cast<int>(platforms.size()))
            throw std::runtime_error("Platform index out of range");
        platform_ = platforms[platform_index];
        auto devices = all_devices(platform_);
        int d = device_index < 0 ? 0 : device_index;
        if (d >= static_cast<int>(devices.size()))
            throw std::runtime_error("Device index out of range");
        device_ = devices[d];
    } else {
        // Auto-pick: prefer the first GPU on any platform, otherwise the first device at all.
        bool found = false;
        for (auto& p : platforms) {
            for (auto& d : all_devices(p)) {
                if (d.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU) {
                    platform_ = p; device_ = d; found = true;
                    break;
                }
            }
            if (found) break;
        }
        for (auto& p : platforms) {
            if (found) break;
            auto devices = all_devices(p);
            if (!devices.empty()) { platform_ = p; device_ = devices[0]; found = true; }
        }
        if (!found)
            throw std::runtime_error("No OpenCL devices found");
    }

    context_ = cl::Context(device_);
    // Profiling must be enabled on the queue, otherwise event timestamps are unavailable.
    queue_ = cl::CommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE);
}

cl::Program algorithm_driver::load_program(const std::string& path, const std::string& options,
                                           double* build_ms)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open kernel file: " + path);
    std::stringstream source;
    source << file.rdbuf();

    cl::Program program(context_, source.str());

    auto start = std::chrono::steady_clock::now();
    try {
        program.build(device_, options.c_str());
    } catch (const cl::BuildError& e) {
        std::string log;
        for (const auto& entry : e.getBuildLog())
            log += entry.second;
        throw std::runtime_error("Failed to build " + path + " with options \"" + options + "\":\n" + log);
    }
    if (build_ms)
        *build_ms = elapsed_ms(start);
    return program;
}

benchmark_result algorithm_driver::benchmark(Algorithm& algo, const benchmark_config& cfg)
{
    benchmark_result r;
    r.name = algo.name();
    r.n = cfg.n;
    r.throughput_unit = algo.throughput_unit();
    r.work_units = algo.work_units(cfg.n);

    std::string options = algo.build_options();
    if (!cfg.extra_options.empty())
        options += " " + cfg.extra_options;
    cl::Program program = load_program(cfg.kernel_dir + "/" + algo.kernel_file(), options, &r.build_ms);

    auto start = std::chrono::steady_clock::now();
    algo.setup(context_, queue_, program, cfg.n);
    queue_.finish();
    r.setup_ms = elapsed_ms(start);

    for (int i = 0; i < cfg.warmup; ++i)
        algo.run(queue_).wait();

    for (int i = 0; i < cfg.iters; ++i) {
        cl::Event ev = algo.run(queue_);
        ev.wait();
        cl_ulong t_start = ev.getProfilingInfo<CL_PROFILING_COMMAND_START>(); // nanoseconds
        cl_ulong t_end = ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        r.kernel_ms.push_back((t_end - t_start) * 1e-6);
    }

    r.verified = algo.verify(queue_);

    if (!r.kernel_ms.empty()) {
        std::vector<double> sorted = r.kernel_ms;
        std::sort(sorted.begin(), sorted.end());
        r.min_ms = sorted.front();
        size_t mid = sorted.size() / 2;
        r.median_ms = sorted.size() % 2 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
        r.mean_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    }
    return r;
}

void algorithm_driver::list_devices()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (size_t p = 0; p < platforms.size(); ++p) {
        std::cout << "Platform " << p << ": " << platforms[p].getInfo<CL_PLATFORM_NAME>() << "\n";
        auto devices = all_devices(platforms[p]);
        for (size_t d = 0; d < devices.size(); ++d) {
            const auto& dev = devices[d];
            std::cout << "  Device " << d << ": " << dev.getInfo<CL_DEVICE_NAME>()
                      << " | " << dev.getInfo<CL_DEVICE_VERSION>()
                      << " | CUs: " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
                      << " | global mem: " << (dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() >> 20) << " MiB"
                      << " | local mem: " << (dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() >> 10) << " KiB"
                      << " | max WG: " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
        }
    }
}

void print_report(const benchmark_result& r, const algorithm_driver& driver)
{
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Device      : " << driver.device().getInfo<CL_DEVICE_NAME>() << "\n"
              << "Algorithm   : " << r.name << "  (n = " << r.n << ")\n"
              << "Build       : " << r.build_ms << " ms\n"
              << "Setup       : " << r.setup_ms << " ms (alloc + upload)\n"
              << "Kernel      : min " << r.min_ms << " ms | median " << r.median_ms
              << " ms | mean " << r.mean_ms << " ms  (" << r.kernel_ms.size() << " iters)\n";
    if (r.work_units > 0 && r.median_ms > 0)
        std::cout << "Throughput  : " << (r.work_units / (r.median_ms * 1e-3)) / 1e9 << " "
                  << r.throughput_unit << " (at median)\n";
    std::cout << "Verify      : " << (r.verified ? "PASS" : "FAIL") << "\n";
}
