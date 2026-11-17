#include "algorithm_driver.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#ifndef KERNEL_DIR
#define KERNEL_DIR "kernels"
#endif

namespace {

void usage(const char* prog)
{
    std::cout << "Usage:\n"
              << "  " << prog << " --list                 list registered algorithms\n"
              << "  " << prog << " --devices              list OpenCL platforms/devices\n"
              << "  " << prog << " <algorithm> [options]\n\n"
              << "Options:\n"
              << "  --size N          problem size (default 16777216)\n"
              << "  --iters K         timed iterations (default 20)\n"
              << "  --warmup W        untimed warmup iterations (default 3)\n"
              << "  --platform P      platform index (default: auto, prefers GPU)\n"
              << "  --device D        device index within the platform (default 0)\n"
              << "  --options \"...\"   extra kernel build options, e.g. \"-DTILE=16\"\n"
              << "  --kernel-dir DIR  directory with .cl files (default " KERNEL_DIR ")\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    std::string first = argv[1];
    if (first == "-h" || first == "--help") { usage(argv[0]); return 0; }
    if (first == "--list") {
        for (const auto& entry : algorithm_registry())
            std::cout << entry.first << "\n";
        return 0;
    }

    try {
        if (first == "--devices") {
            algorithm_driver::list_devices();
            return 0;
        }

        benchmark_config cfg;
        cfg.kernel_dir = KERNEL_DIR;
        int platform_index = -1, device_index = -1;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--size")            cfg.n = std::stoull(next());
            else if (arg == "--iters")      cfg.iters = std::stoi(next());
            else if (arg == "--warmup")     cfg.warmup = std::stoi(next());
            else if (arg == "--platform")   platform_index = std::stoi(next());
            else if (arg == "--device")     device_index = std::stoi(next());
            else if (arg == "--options")    cfg.extra_options = next();
            else if (arg == "--kernel-dir") cfg.kernel_dir = next();
            else throw std::runtime_error("Unknown option: " + arg);
        }

        auto it = algorithm_registry().find(first);
        if (it == algorithm_registry().end()) {
            std::cerr << "Unknown algorithm '" << first << "'. Use --list.\n";
            return 1;
        }
        auto algo = it->second();

        algorithm_driver driver(platform_index, device_index);
        benchmark_result result = driver.benchmark(*algo, cfg);
        print_report(result, driver);
        return result.verified ? 0 : 2;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error in " << e.what() << ": " << e.err() << " (" << cl_error_string(e.err()) << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 1;
}
