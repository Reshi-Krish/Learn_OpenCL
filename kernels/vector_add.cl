// C = A + B, one work-item per element.
// `n` guards the tail so the global size may be rounded up to a multiple of the work-group size.
__kernel void vector_add(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const uint n)
{
    const size_t i = get_global_id(0);
    if (i < n)
        C[i] = A[i] + B[i];
}
