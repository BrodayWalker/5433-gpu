#include <stdio.h>
#include <cuda.h>

int main()
{
    int nDevices;

    // Get number of CUDA devices
    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("=============================\n");

        // Device number and name
        printf("Device Number: %d\n", i);
        printf("Device name: %s\n", prop.name);

        // Size of shared memory per block
        printf("Size of shared memory per block: %zu\n", prop.sharedMemPerBlock);

        // Number of registers per block
        printf("Number of registers per block: %d\n", prop.regsPerBlock);

        // Warp size
        printf("Warp size: %d\n", prop.warpSize);

        // Maximum number of threads per block
        printf("Maximum number of threads per block: %d\n", 
            prop.maxThreadsPerBlock);

        // Maximum number of threads for 3D layout (X, Y, Z)
        printf("Maximum number of threads for 3d layout (X, Y, Z): %d, %d, %d\n", 
            prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

        // Maximum grid size (X, Y, Z)
        printf("Maximum grid size (X, Y, Z): %d, %d, %d\n", prop.maxGridSize[0],
            prop.maxGridSize[1], prop.maxGridSize[2]);

        // Maximum number of blocks per streaming processor (SM)

    }
    printf("=============================\n");

    return 0;
}