#include <stdio.h>
#include <complex.h>
#include <tgmath.h>

int main()
{
    // Test
    double complex zero = 3.6 + 2.6 * I;
    double complex other = 1 - 0 * I;
    double complex diff = zero * other;

    printf("Answer: %.2f %+.2fi\n", creal(diff), cimag(diff));

    return 0;
}