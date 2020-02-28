// nvcc BrodayWalker1B.cu -o BrodayWalker1B.exe

#include <stdio.h>
#include <cuda.h>

enum N {N = 32};

void print(int [], int);

__global__ void matmulKernel(int *Ad, int *Bd, int *Cd, int width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0;

    for (int k = 0; k < width; k++)
    {
        int Aelement = Ad[ty * width + k];
        int Belement = Bd[k * width + tx];
        sum += Aelement * Belement;
    }

    Cd[ty * width + tx] = sum;
}

int main()
{
    // Declarations
    int A[N * N], B[N * N], C[N * N];
    int *Ad, *Bd, *Cd;
    int size = N * N * sizeof(int);

    // Declare the timer
    // Reference: 
    // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Fill arrays A and C
    // Array C will be filled with 0s
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = i;
            C[i * N + j] = 0;
        }
    
    // Fill B
    int row = N - 1;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            B[i * N + j] = row;
        row--;        
    }

    /* Allocate memory and copy to device */
    cudaMalloc((void**)&Ad, size);
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Bd, size);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Cd, size);
    /* End memory allocation and copying to device */
    
    /* Define grid and block dimensions */
    dim3 dimGrid( 1, 1 );
    dim3 dimBlock( N, N );

    /* Record start time */
    cudaEventRecord(start);
    /* Invoke the kernel */
    matmulKernel<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, N);
    /* Record end time */
    cudaEventRecord(stop);

    /* Copy the matrix multiplication results from device to host */
    cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

    /* Block CPU execution until the specified event is recorded */
    cudaEventSynchronize(stop);
    /* Returns the elapsed time in milliseconds to the first argument */
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    // Print results
    printf("\n\nArray C: \n");
    print(C, N);
    
    // Print elapsed time
    printf("\nElapsed time in milliseconds: %f\n", milliseconds);

    return 0;
}

// function: print()
// parameters: int [][N], int width
// The print function prints a 2D array
void print(int ray[], int width)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
            printf("%d ", ray[i * width + j]);
        printf("\n");
    }
}