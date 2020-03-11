//***************************************************************************
//  Name: Broday Walker
//  
//  Links: 
//  1. https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/
//  2. https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__DEVICE_g5aa4f47938af8276f08074d09b7d520c.html#g5aa4f47938af8276f08074d09b7d520c
//  3. https://devtalk.nvidia.com/default/topic/461911/tesla-c1060-max-blocks-per-streaming-multiprocessor/
//  4. https://stackoverflow.com/questions/502856/whats-the-difference-between-size-t-and-int-in-c
//
//  To run for the gtx queue: sbatch gtxProperties
//  To run for the v100 queue: sbatch voltaProperties
//***************************************************************************

#include <stdio.h>
#include <cuda.h>

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=============================\n");

    // Device number and name
    printf("Device number: %d\n", 0);
    printf("Device name: %s\n", prop.name);

    // Size of shared memory per block
    printf("Size of shared memory per block: %zu\n", 
        prop.sharedMemPerBlock);

    // Number of registers per block
    printf("Number of registers per block: %d\n", prop.regsPerBlock);

    // Warp size
    printf("Warp size: %d\n", prop.warpSize);

    // Maximum number of threads per block
    printf("Maximum number of threads per block: %d\n", 
        prop.maxThreadsPerBlock);

    // Maximum number of threads for 3D layout (X, Y, Z)
    printf("Maximum number of threads for 3d layout (X, Y, Z): "
        "%d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], 
        prop.maxThreadsDim[2]);

    // Maximum grid size (X, Y, Z)
    printf("Maximum grid size (X, Y, Z): %d, %d, %d\n", 
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // Maximum number of blocks per streaming processor (SM)
    printf("Maximum number of blocks per streaming processor: "
        "not given\n");

    printf("=============================\n");

    return 0;
}