#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include<CL/opencl.hpp>
#include<iostream>

#define SIZE 10
int main()
{
    // Querying all available platforms and choosing default platform
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No OpenCL platforms found.\n";
        exit(1);
    }

    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " <<default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    // Querying all available devices and choosing default device
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found.\n";
        exit(1);
    }

    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    cl::Context context({default_device});

    int A_h[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int B_h[] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

    cl::Buffer A_d(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE);
    cl::Buffer B_d(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE);
    cl::Buffer C_d(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE);

    cl::CommandQueue queue(context, default_device);

    queue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(int) * SIZE, A_h);
    queue.enqueueWriteBuffer(B_d, CL_TRUE, 0, sizeof(int) * SIZE, B_h);


    std::string kernel_code =
    "   void kernel simple_add(global const int* A, global const int* B, global int* C){ "
    "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
    "   } ";

    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    cl::Program program(context, sources);

    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }


    cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> simple_add(cl::Kernel(program, "simple_add"));
    cl::NDRange global(SIZE);
    simple_add(cl::EnqueueArgs(queue, global), A_d, B_d, C_d).wait();

    int C_h[SIZE];
    //read result C_d from the device to array C_h
    queue.enqueueReadBuffer(C_d, CL_TRUE, 0, sizeof(int) * SIZE, C_h);

    std::cout << " result: \n";
    for (int i = 0; i<10; i++) {
        std::cout << C_h[i] << " ";
    }
    std::cout<<std::endl;

    return 0;

}