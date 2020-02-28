#include <stdio.h>
#include <complex.h>
#include <math.h>

int main()
{
    enum N {N = 10};
    // Size of double complex data type
    const int D_COMPLEX_SIZE = N * sizeof(double complex);

    printf("Double complex size: %d\n", D_COMPLEX_SIZE);


    return 0;
}