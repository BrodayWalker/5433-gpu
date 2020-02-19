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

    // Print the array to make sure it is correct
    for (int i = 0; i < N; i++)
        printf("Data %i: %.2f %+.2fi\n", i, creal(data[i]), cimag(data[i]));

    // Sum the even component
    for(int i = 0; i < (N / 2); i++)
    {
        double even_index = (i * 2.0) / N;
        double odd_index = ((i * 2.0) + 1) / N;
        double complex even = data[i * 2] * (ccos(TAU * even_index) - I * csin(TAU * even_index));
        double complex odd = data[(i * 2) + 1] * (ccos(TAU * odd_index) - I * csin(TAU * odd_index));
        printf("Even[%d]: %f %+fi\n", i, creal(even), cimag(even));
        printf("Odd[%d]: %f %+fi\n", i, creal(odd), cimag(odd));
        
        even_sum += even;
        odd_sum += odd;  
    }

    // Print the sum
    printf("Sum of the even part: %.2f %+.2fi\n", creal(even_sum), cimag(even_sum));
    printf("Sum of the odd part: %.2f %+.2fi\n", creal(odd_sum), cimag(odd_sum));
    

    return 0;
}