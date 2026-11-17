#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS   // optional: turns CL errors into cl::Error exceptions

#pragma once

#include <CL/opencl.hpp>
#include<vector>
#include<string>

struct algorithm_params
{
    std::vector<cl::Buffer> inputBuffers, outputBuffers;

};
class algorithm_driver
{

private:
    
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue work_queue;
    cl::Program algorithm;
    algorithm_params params;

public:
    void init_driver();
    void benchmark_algorithm();
    void load_algorithm();
};