// gcc BrodayWalkerSequential.c -o BrodayWalkerSequential.exe

#include <stdio.h>
#include <sys/time.h> // For timing

enum N {N = 32};

void print(int [][N], int);

int main()
{
    // Declarations
    int A[N][N], B[N][N], C[N][N];
    int sum = 0;

    // Timers
    struct timeval start, end;

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

    gettimeofday(&start, NULL);
    // Matrix multiplication - This method assumes all elements in array
    // C are initialized to 0.
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
              C[i][j] += A[i][k] * B[k][j];
    
    gettimeofday(&end, NULL);
    // Calculate function running time
    double seconds = (end.tv_sec - start.tv_sec);
	double micros = ((seconds * 1000000.0) + end.tv_usec) - (start.tv_usec);
    // Convert from microseconds to milliseconds
    double millis = micros / 1000.0;
    
	// Sum the array
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += C[i][j];

    printf("The summation of all the elements is %d.\n\n", sum);

    printf("Time elpased is %f seconds and %f milliseconds.\n", seconds, millis);

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