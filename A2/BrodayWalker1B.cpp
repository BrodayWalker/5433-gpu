// gcc BrodayWalkerSequential.c -o BrodayWalkerSequential.exe

#include <stdio.h>
#include <cuda.h>

enum N {N = 32};

void print(int [][N], int);

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
    int A[N][N], B[N][N], C[N][N];
    int *Ad, Bd, Cd;
    int size = N * N * sizeof(int);

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


    cudaMalloc((void**)&Ad, size);
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Bd, size);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Cd, size);
    
    dim3 dimGrid( 1, 1 );
    dim3 dimBlock( N, N );

    matmulKernel<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, N);

    cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);


    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    // Print array A
    printf("Array A: \n");
    print(A, N);

    printf("\n\nArray B: \n");
    print(B, N);

    printf("\n\nArray C: \n");
    print(C, N);
    
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