#pragma once

#include "cl_common.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>

// One benchmarkable algorithm. The driver owns the device/context/queue and the
// timing loop; the algorithm owns its kernel, buffers, and correctness check.
struct Algorithm {
    virtual ~Algorithm() = default;
    virtual std::string name() const = 0;
    virtual std::string kernel_file() const = 0;            // relative to the kernel dir, e.g. "matmul.cl"
    virtual std::string build_options() const { return ""; } // e.g. "-DTILE=16"
    virtual void setup(cl::Context&, cl::CommandQueue&, cl::Program&, size_t n) = 0; // alloc + upload + setArg
    virtual cl::Event run(cl::CommandQueue&) = 0;           // enqueue kernel, return its event
    virtual bool verify(cl::CommandQueue&) = 0;             // read back + compare with a CPU reference
    virtual double work_units(size_t /*n*/) const { return 0; } // FLOPs or bytes moved by one run()
    virtual std::string throughput_unit() const { return "GB/s"; } // "GB/s" for bytes, "GFLOP/s" for FLOPs
};

// ---- Registry: lets main() create an algorithm from its name on the command line ----

using algorithm_factory = std::function<std::unique_ptr<Algorithm>()>;

inline std::map<std::string, algorithm_factory>& algorithm_registry()
{
    static std::map<std::string, algorithm_factory> registry;
    return registry;
}

struct algorithm_registrar {
    algorithm_registrar(const std::string& name, algorithm_factory factory)
    {
        algorithm_registry()[name] = std::move(factory);
    }
};

// Put this at the bottom of an algorithm's .cpp file:  REGISTER_ALGORITHM("vector_add", vector_add)
// (Works because algorithm .cpp files are compiled straight into the executable.
//  If you ever move them into a static library, the linker may drop these objects.)
#define REGISTER_ALGORITHM(NAME, CLASS) \
    static algorithm_registrar CLASS##_registrar_(NAME, [] { return std::make_unique<CLASS>(); })
