//***************************************************************************
//  Assignment #1
//  Name: Broday Walker
//  Parallel Programming
//  Date: 
//***************************************************************************
//  Place general documentation here.
//
//
//***************************************************************************

/* FUNCTION TEMPLATE */
// function Name::MethodName()
// Parameters: 
// Discuss what method does


#include <stdio.h>
#include <math.h> // For cos, sin
#include <complex.h>

// C will not let N be declared as a const int and used to declare
// the size of an array for some reason.
#define N 8
#define TAU 6.28318530718 // 2pi

void print_array(const double *);

int main()
{
    // Every element after element 7 (the 8th value) will be initialized
    // to 0.
    double real[N] = {3.6, 2.9, 5.6, 4.8, 3.3, 5.9, 5.0, 4.3};
    double imaginary[N] = {2.6, 6.3, 4, 9.1, 0.4, 4.8, 2.6, 4.1};
    double X_even[N] = {0};
    double X_odd[N] = {0};
    double XR[N] = {0};
    double XI[N] = {0};
    
    printf("i: i%f\n", _Complex_I);
    
    double even_component_sumR = 0;
    double even_component_sumI = 0;
    for (int i = 1; i <= 1; i++)
    {
        for (int j = 0; j < N / 2; j++)
        {
            // Even side
            int even = 2 * j;
            // Odd side
            int odd = 2 * j + 1;

            // Even euler
            double even_eulerR = cos(TAU * real[j]);
            double even_eulerI = sin(TAU * imaginary[j]);
            even_component_sumR += even_eulerR;
            even_component_sumI += even_eulerI;

        }
        printf("Even real sum: %f\n Even imaginary sum: %f", 
            even_component_sumR, even_component_sumI);
        
    }

    printf("%f", TAU);
    


    return 0;
}

void print_array(const double *ray)
{
    // Print the array
    for (int i = 0; i < N; i++)
    {
        printf("%.2f ", ray[i]);
    }
    printf("\n");
}

