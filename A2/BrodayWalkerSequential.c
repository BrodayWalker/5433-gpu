// gcc BrodayWalkerSequential.c -o BrodayWalkerSequential.exe

#include <stdio.h>
#include <cuda.h> // For timing

enum N {N = 32};

void print(int [][N], int);

int main()
{
    // Declarations
    int A[N][N], B[N][N], C[N][N];

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
            A[i][j] = i;
            C[i][j] = 0;
        }
    
    // Fill B
    int row = N - 1;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            B[i][j] = row;
        row--;        
    }

    // Print arrays
    printf("Array A: \n");
    print(A, N);

    printf("\n\nArray B: \n");
    print(B, N);

    printf("\n\nArray C before matrix multiplication: \n");
    print(C, N);

    /* Record start time */
    cudaEventRecord(start);
    // Matrix multiplication - This method assumes all elements in array
    // C are initialized to 0.
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
              C[i][j] += A[i][k] * B[k][j];
            
    /* Record end time */
    cudaEventRecord(stop);
    /* Returns the elapsed time in milliseconds to the first argument */
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\n\nArray C after matrix multiplication: \n");
    print(C, N);

    // Print elapsed time
    printf("\nElapsed time in milliseconds: %f\n", milliseconds);

    return 0;
}

// function: print()
// parameters: int [][N], int width
// The print function prints a 2D array
void print(int ray[][N], int width)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
            printf("%d ", ray[i][j]);
        printf("\n");
    }
}