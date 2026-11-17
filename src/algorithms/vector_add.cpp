#include "algorithm.hpp"

#include <cmath>
#include <random>
#include <vector>

// Reference plugin: copy this file to add a new algorithm (e.g. matmul.cpp + kernels/matmul.cl).
class vector_add : public Algorithm
{
public:
    std::string name() const override { return "vector_add"; }
    std::string kernel_file() const override { return "vector_add.cl"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, cl::Program& program, size_t n) override
    {
        n_ = n;
        a_.resize(n);
        b_.resize(n);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < n; ++i) {
            a_[i] = dist(rng);
            b_[i] = dist(rng);
        }

        const size_t bytes = n * sizeof(float);
        a_d_ = cl::Buffer(context, CL_MEM_READ_ONLY, bytes);
        b_d_ = cl::Buffer(context, CL_MEM_READ_ONLY, bytes);
        c_d_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, bytes);
        queue.enqueueWriteBuffer(a_d_, CL_FALSE, 0, bytes, a_.data());
        queue.enqueueWriteBuffer(b_d_, CL_FALSE, 0, bytes, b_.data());

        kernel_ = cl::Kernel(program, "vector_add");
        kernel_.setArg(0, a_d_);
        kernel_.setArg(1, b_d_);
        kernel_.setArg(2, c_d_);
        kernel_.setArg(3, static_cast<cl_uint>(n));
    }

    cl::Event run(cl::CommandQueue& queue) override
    {
        cl::Event ev;
        // Local size NullRange: let the driver choose the work-group size.
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, cl::NDRange(n_), cl::NullRange, nullptr, &ev);
        return ev;
    }

    bool verify(cl::CommandQueue& queue) override
    {
        std::vector<float> c(n_);
        queue.enqueueReadBuffer(c_d_, CL_TRUE, 0, n_ * sizeof(float), c.data());
        for (size_t i = 0; i < n_; ++i)
            if (std::fabs(c[i] - (a_[i] + b_[i])) > 1e-6f)
                return false;
        return true;
    }

    // Memory-bound: 2 reads + 1 write of n floats per run.
    double work_units(size_t n) const override { return 3.0 * n * sizeof(float); }
    std::string throughput_unit() const override { return "GB/s"; }

private:
    size_t n_ = 0;
    std::vector<float> a_, b_;
    cl::Buffer a_d_, b_d_, c_d_;
    cl::Kernel kernel_;
};

REGISTER_ALGORITHM("vector_add", vector_add);
