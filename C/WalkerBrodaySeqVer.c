//***************************************************************************
// Assignment #1
// Name: Broday Walker
// Parallel Programming
// Date: February 21, 2020
//***************************************************************************
// This program implements a serial version of the Cooley-Tukey Fast Fourier
// Transform algorithm. Per the program specification, an array of complex
// numbers is stored with their respective predetermined values. Using the
// Cooley-Tukey algorithm, N Fourier coefficients are calculated and printed
// to a file. 
//
// To use this program, first compile it with a GNU compiler.
// Example: gcc WalkerBrodaySeqVer.c -o fft.exe
// The above command compiles this code into an executable named fft.exe.
// To run fft.exe from the command line, navigate to the folder containing
// the executable use ./fft.exe
// Running the program creates an output file which is stored in the same
// folder the executable was run from. This file contains the calculations
// found by the Cooley-Tukey algorithm. 
//***************************************************************************

#include <stdio.h>
#include <complex.h>
#include <math.h>

#define TAU 6.28318530718 // 2pi

int main()
{
    // Use enum N {N = <some value>} because enum members are constant 
    // expressions in C.
    // Using const int N = <some number> does not work in C as constant
    // variables are really just read-only variables that can change their
    // value.
    enum N {N = 10};
    // An array of complex numbers per the specification
    double complex data[N] = {3.6 + 2.6 * I, 2.9 + 6.3 * I, 5.6 + 4.0 * I, 
        4.8 + 9.1 * I, 3.3 + 0.4 * I, 5.9 + 4.8 * I, 5.0 + 2.6 * I, 
        4.3 + 4.1 * I};
    // The results array
    double complex X[N] = {0 + 0 * I};
    // Use these variables to store intermediate computation
    double complex even_sum = 0 + 0 * I;
    double complex odd_sum = 0 + 0 * I;
    

    // Print the array to make sure it is correct
    for (int i = 0; i < N; i++)
        printf("Data %i: %.2f %+.2fi\n", i, creal(data[i]), cimag(data[i]));

    // Two for-loops make up the Cooley-Tukey algorithm. The outside loop
    // controls how many Fourier coefficients are calculated. For each
    // coefficient to be calculated, the inner loop runs N/2 times, finding
    // both the even- and odd-indexed parts for Xm. Once the inner loop
    // has completed, the results are aggregated and the calculation for
    // a Fourier coefficient has completed.
    for (int k = 0; k < N; k++) 
    {
        for(int j = 0; j < (N / 2); j++)
        {
            double even_index = (j * 2.0) / N;
            double odd_index = ((j * 2.0) + 1) / N;
            double complex even = data[j * 2] * (ccos(TAU * even_index) - I * csin(TAU * even_index * k));
            double complex odd = data[(j * 2) + 1] * (ccos(TAU * odd_index) - I * csin(TAU * odd_index* k));
            // Print for testing
            printf("Even[%d]: %f %+fi\n", j, creal(even), cimag(even));
            printf("Odd[%d]: %f %+fi\n", j, creal(odd), cimag(odd));

            even_sum += even;
            odd_sum += odd;  
        }

        // Print the sum of the even and odd parts
        printf("Sum of the even part: %f %+fi\n", creal(even_sum), cimag(even_sum));
        printf("Sum of the odd part: %f %+fi\n", creal(odd_sum), cimag(odd_sum));

        // Combine all real and imaginary numbers to get X[k]
        X[k] += even_sum + odd_sum;
    }

    // Output the results
    printf("TOTAL PROCESSED SAMPLES: %i\n", N);
    printf("================================\n");
    // Print X, the results array
    for (int i = 0; i < N; i++)
    {
        printf("XR[%i]: %.4f\tXI[%i]: %.4fi\n", i, creal(X[i]), i, cimag(X[i]));
        printf("================================\n");
    }
    
    return 0;
}