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
    cl::CommandQueue work_queue;
    cl::Program algorithm;
    algorithm_params params;

public:
    void init_driver();
    void benchmark_algorithm();
    void load_algorithm(std::string algorithm);
};