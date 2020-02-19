#include <stdio.h>
#include <complex.h>
#include <math.h>

#define TAU 6.28318530718 // 2pi

int main()
{
    // Use enum N {N = some value} because enum members are constant 
    // expressions.
    // Using const in N = some number does not work in C as constant
    // variables are really just read-only variables that can change their
    // value.
    enum N {N = 8};
    // An array of complex numbers per the specification
    double complex data[N] = {3.6 + 2.6 * I, 2.9 + 6.3 * I, 5.6 + 4.0 * I, 
        4.8 + 9.1 * I, 3.3 + 0.4 * I, 5.9 + 4.8 * I, 5.0 + 2.6 * I, 
        4.3 + 4.1 * I};
    double complex even_sum = 0 + 0 * I;
    double complex odd_sum = 0 + 0 * I;
    double complex k_sum = 0 + 0 * I;

    // Print the array to make sure it is correct
    for (int i = 0; i < N; i++)
        printf("Data %i: %.2f %+.2fi\n", i, creal(data[i]), cimag(data[i]));

    
    for (int k = 1; k <= 1; k++) 
    {
        for(int j = 0; j < (N / 2); j++)
        {
            double even_index = (j * 2.0) / N;
            double odd_index = ((j * 2.0) + 1) / N;
            double complex even = data[j * 2] * (ccos(TAU * even_index) - I * csin(TAU * even_index * k));
            double complex odd = data[(j * 2) + 1] * (ccos(TAU * odd_index) - I * csin(TAU * odd_index* k));
            printf("Even[%d]: %f %+fi\n", j, creal(even), cimag(even));
            printf("Odd[%d]: %f %+fi\n", j, creal(odd), cimag(odd));

            even_sum += even;
            odd_sum += odd;  
        }

        // Print the sum of the even and odd parts
        printf("Sum of the even part: %f %+fi\n", creal(even_sum), cimag(even_sum));
        printf("Sum of the odd part: %f %+fi\n", creal(odd_sum), cimag(odd_sum));

        // Combine all real and imaginary numbers to get X[k]
        k_sum += even_sum + odd_sum;
    }
    // Print X(1)
    printf("X(1): %f %fi\n", creal(k_sum), cimag(k_sum));
    
    

    return 0;
}