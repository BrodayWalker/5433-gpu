// On Maverick2: sbatch mvk2SeqMatMul
// gcc BrodayWalkerSequential.c -o BrodayWalkerSequential.exe

//***************************************************************************
//  Name: Broday Walker
//  Instructor: Dr. Colmenares
//  Class: CMPS 5433
//  Date: March 2, 2020
//***************************************************************************
//  This program implements sequential matrix multiplication using the
// traditional, straight-forward O(n^3) algorithm. When the program is run,
// the runtime of the matrix multiplication algorithm is calculated and
// printed in milliseconds. The sum of all elements in the array is also
// given.
//***************************************************************************

#include <stdio.h>	/* for printf */
#include <stdint.h>	/* for uint64 definition */
#include <stdlib.h>	/* for exit() definition */
#include <time.h>	/* for clock_gettime */

#define BILLION 1000000000L
#define MILLION 1000000

enum N {N = 32}; // Define size of array

int main()
{
    // Declarations
    int A[N][N], B[N][N], C[N][N];
    int sum = 0;

    // Timers
    //uint64_t diff;
    double diff;
	struct timespec start, end;

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

    /* measure monotonic time */
	clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
    // Matrix multiplication - This method assumes all elements in array
    // C are initialized to 0.
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
              C[i][j] += A[i][k] * B[k][j];
    
    clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

    // Get time in milliseconds
    diff = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
	// Sum the array
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += C[i][j];

    // Print results
    printf("The summation of all the elements is %d.\n\n", sum);

    printf("Time elpased is %f milliseconds.\n",  diff);

    return 0;
}