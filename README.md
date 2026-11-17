# Learn_OpenCL

A small harness for learning OpenCL. You write a kernel and a short C++ class
that sets it up and checks the result. The harness picks a device, compiles the
kernel, times it on the GPU and prints a report:

```
$ ./build/bench vector_add --size 16777216
Device      : Intel(R) Iris(R) Xe Graphics [0x9a49]
Algorithm   : vector_add  (n = 16777216)
Build       : 208.471 ms
Setup       : 370.489 ms (alloc + upload)
Kernel      : min 10.686 ms | median 11.786 ms | mean 11.886 ms  (20 iters)
Throughput  : 17.082 GB/s (at median)
Verify      : PASS
```

## Requirements

- An OpenCL driver for your device. Check with `clinfo -l`.
- OpenCL headers (`opencl-c-headers`, `opencl-clhpp-headers`) and the ICD loader (`ocl-icd-opencl-dev`).
- CMake ≥ 3.16 and a C++17 compiler.

## Build and run

```bash
cmake -S . -B build
cmake --build build -j

./build/bench --devices                 # list platforms/devices
./build/bench --list                    # list registered algorithms
./build/bench vector_add                # run with defaults
./build/bench vector_add --size 1048576 --iters 50 --warmup 5
./build/bench vector_add --options "-cl-fast-relaxed-math"
./build/bench vector_add --platform 0 --device 0

./build/basic_vectoradd                 # standalone example, no harness
```

| Option | Meaning | Default |
|---|---|---|
| `--size N` | Problem size. Each algorithm decides what `N` means. | 16777216 |
| `--iters K` | Timed iterations | 20 |
| `--warmup W` | Untimed iterations run first | 3 |
| `--platform P` / `--device D` | Choose a device by index | auto (first GPU) |
| `--options "..."` | Extra kernel build options, added after the algorithm's own | none |
| `--kernel-dir DIR` | Where `.cl` files are read from | `<repo>/kernels` |

`bench` exits with `0` on success, `2` if verification fails and `1` on any other error.

## Repository layout

```
CMakeLists.txt              builds `bench` and `basic_vectoradd`
include/
  cl_common.hpp             CL_HPP_* version/exception macros + <CL/opencl.hpp>; include this, not opencl.hpp
  algorithm.hpp             the Algorithm interface + REGISTER_ALGORITHM registry
  algorithm_driver.hpp      device/context/queue setup, program loading, timing loop, report
src/
  main.cpp                  command-line parsing; finds the algorithm by name and runs it
  algorithm_driver.cpp
  cl_common.cpp             cl_error_string(): error code -> name
  algorithms/               one .cpp per algorithm (picked up automatically by CMake)
    vector_add.cpp
  basic_vectoradd.cpp       standalone single-file example of the raw OpenCL flow
kernels/                    OpenCL C kernels (.cl), compiled at runtime
  vector_add.cl
```

### How a run works

```
main.cpp
  └─ algorithm_registry()["vector_add"]()          -> creates the Algorithm object
  └─ algorithm_driver(platform, device)            -> picks a device, creates the context and a profiling queue
  └─ driver.benchmark(algo, cfg)
       ├─ load_program(kernel_dir/kernel_file(),     -> reads the .cl file and builds it for the device
       │               build_options() + --options)     (build time is reported)
       ├─ algo.setup(context, queue, program, n)    -> allocate buffers, upload data, create kernel, setArg
       ├─ algo.run(queue).wait()  × warmup
       ├─ algo.run(queue)         × iters           -> GPU time from the returned event's profiling timestamps
       └─ algo.verify(queue)                        -> read back the result and compare with a CPU reference
  └─ print_report(result)
```

Responsibilities are split like this:

- **The driver** owns the device, context and command queue. It also compiles the kernel and does all the timing.
- **Each algorithm** owns its data, buffers, `cl::Kernel` and correctness check. It never times anything itself.

## Adding a new algorithm

Here we add `matmul`. Use [src/algorithms/vector_add.cpp](src/algorithms/vector_add.cpp)
as a complete working example.

### 1. Write the kernel: `kernels/matmul.cl`

```c
// C = A * B, all N x N, row-major. One work-item per element of C.
__kernel void matmul(__global const float* A,
                     __global const float* B,
                     __global float* C,
                     const uint N)
{
    const size_t row = get_global_id(1);
    const size_t col = get_global_id(0);
    if (row >= N || col >= N) return;
    // TODO: C[row * N + col] = dot(row of A, column of B)
}
```

The kernel name (`matmul`) must match the name passed to `cl::Kernel(program, "...")` in step 2.

### 2. Write the plugin: `src/algorithms/matmul.cpp`

```cpp
#include "algorithm.hpp"

#include <cmath>
#include <random>
#include <vector>

class matmul : public Algorithm
{
public:
    std::string name() const override { return "matmul"; }
    std::string kernel_file() const override { return "matmul.cl"; }
    // std::string build_options() const override { return "-DTILE=16"; }   // optional

    void setup(cl::Context& context, cl::CommandQueue& queue, cl::Program& program, size_t n) override
    {
        n_ = n;                                   // here n = matrix dimension N
        // TODO: fill a_, b_ (N*N floats) with random values
        // TODO: create a_d_, b_d_ (CL_MEM_READ_ONLY) and c_d_ (CL_MEM_WRITE_ONLY), enqueueWriteBuffer a_ and b_
        kernel_ = cl::Kernel(program, "matmul");
        // TODO: kernel_.setArg(0..3, ...)  — the last argument is static_cast<cl_uint>(n)
    }

    cl::Event run(cl::CommandQueue& queue) override
    {
        cl::Event ev;
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, cl::NDRange(n_, n_), cl::NullRange, nullptr, &ev);
        return ev;                                // the driver reads the GPU time from this event
    }

    bool verify(cl::CommandQueue& queue) override
    {
        // TODO: enqueueReadBuffer(c_d_, CL_TRUE, ...) and compare with a CPU matmul.
        //       Use a relative tolerance (e.g. 1e-3): float sums are accumulated in a different order on the GPU.
        return false;
    }

    double work_units(size_t n) const override { return 2.0 * n * n * n; }   // FLOPs per run
    std::string throughput_unit() const override { return "GFLOP/s"; }

private:
    size_t n_ = 0;
    std::vector<float> a_, b_;
    cl::Buffer a_d_, b_d_, c_d_;
    cl::Kernel kernel_;
};

REGISTER_ALGORITHM("matmul", matmul);
```

### 3. Build and run

```bash
cmake --build build                       # CMake finds the new .cpp in src/algorithms/ automatically
./build/bench --list                      # should now show matmul
./build/bench matmul --size 1024 --iters 10
```

### Checklist / conventions

- **Header:** include `algorithm.hpp` (it includes `cl_common.hpp`), never `<CL/opencl.hpp>` directly. Every file must see the same `CL_HPP_*` macros.
- **Registration:** put `REGISTER_ALGORITHM("<name>", <Class>);` at the bottom of the `.cpp`. The string is what you type on the command line.
- **`kernel_file()`:** a path relative to `kernels/`.
- **`setup()`:** do all allocation, uploading and `setArg` calls here, not in `run()`. Otherwise you are timing setup work.
- **`run()`:** only enqueue the kernel, and return its `cl::Event`. Don't call `finish()` or `wait()`; the driver does that.
- **`work_units()` / `throughput_unit()`:** return bytes per run with `"GB/s"` for memory-bound kernels, or FLOPs per run with `"GFLOP/s"` for compute-bound ones. Throughput = `work_units / median time / 1e9`.
- **`verify()`:** always compare against a simple CPU version. A fast wrong kernel is worthless.
- **Kernel variants:** use `#ifdef` / macros in the `.cl` file. Select them with `build_options()` or `--options "-DTILE=32"`, so one kernel file can cover several variants without recompiling the C++.

## Tips

- **Edit kernels without rebuilding.** Kernels are compiled when `bench` runs, from `<repo>/kernels`. Edit a `.cl` file and re-run `bench`. You only need `cmake --build` after C++ changes.
- **Build errors show the driver's log.** It includes the line and column inside the `.cl` file.
- **OpenCL errors are printed by name.** They appear as `OpenCL error in clXxx: -N (CL_...)`. Exceptions are on (`CL_HPP_ENABLE_EXCEPTIONS`), so you don't need to check return codes by hand.
- **Timing measures kernel execution only.** The `Kernel` line comes from `CL_PROFILING_COMMAND_START/END`. Host-to-device transfers are counted under `Setup`, and compile time under `Build`.
- **Why `CL_HPP_MINIMUM_OPENCL_VERSION` is 200:** the `opencl.hpp` shipped with Ubuntu 22.04 fails to compile with `TARGET=300` + `MINIMUM=120`. See the comment in [include/cl_common.hpp](include/cl_common.hpp).
