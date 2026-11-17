#pragma once

// Every translation unit must see the same CL_HPP_* settings, so they live here
// and every file includes this header instead of <CL/opencl.hpp> directly.
#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif
// 200 (not 120): the opencl.hpp shipped with Ubuntu 22.04 fails to compile with
// TARGET >= 200 && MINIMUM == 120 (it calls an undefined getContextPlatformVersion).
// All current Intel/AMD/NVIDIA drivers are OpenCL 3.0, so requiring 2.0 APIs is fine.
#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#endif
#ifndef CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS   // CL errors throw cl::Error (and cl::BuildError on build failure)
#endif

#include <CL/opencl.hpp>

// Human-readable name for an OpenCL error code, e.g. -5 -> "CL_OUT_OF_RESOURCES".
const char* cl_error_string(cl_int err);
