#include <iostream>
#include <cmath>
#include <complex>

using namespace std;

int main()
{
    const int N = 10;
    
    // An array of complex numbers per the specification
    complex<double> data[N] = {3.6 + 2.6 * 1i, 2.9 + 6.3 * 1i, 5.6 + 4.0 * 1i, 
        4.8 + 9.1 * 1i, 3.3 + 0.4 * 1i, 5.9 + 4.8 * 1i, 5.0 + 2.6 * 1i, 
        4.3 + 4.1 * 1i};

    for (int i = 0; i < N; i++)
        cout << data[i] << '\n';

    return 0;
}