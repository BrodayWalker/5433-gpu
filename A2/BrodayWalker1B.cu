// gcc BrodayWalkerSequential.c -o BrodayWalkerSequential.exe

#include <stdio.h>

enum N {N = 32};

void print(int [][N], int);

int main()
{
    // Declarations
    int A[N][N], B[N][N], C[N][N];

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