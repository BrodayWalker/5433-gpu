#include <iostream>
#include <fstream>
#include <cuda.h>
#include <complex>
#include <thrust/complex.h>
#include <cuComplex.h>

using namespace std;

__global__ void fft(thrust::complex<float> *, thrust::complex<float> *);

int main()
{
    const int N = 10;
    
    // An array of complex numbers per the specification
    complex<double> data[N] = {3.6 + 2.6 * 1i, 2.9 + 6.3 * 1i, 5.6 + 4.0 * 1i, 
       4.8 + 9.1 * 1i, 3.3 + 0.4 * 1i, 5.9 + 4.8 * 1i, 5.0 + 2.6 * 1i, 
        4.3 + 4.1 * 1i};
    
    // The results array, initialized to 0 + 0i for each element
    complex<double> results[N];
    
    // An output file for FFT calculations
    ofstream outfile;
    outfile.open("output.txt");

    // Device pointers
    thrust::complex<float> *datad;
    thrust::complex<float> *resultsd;

    // Size of double complex data type
    const int COMPLEX_ARRAY_SIZE = N * sizeof(cuComplex);

    cudaMalloc( (void**)&datad, COMPLEX_ARRAY_SIZE );
    cudaMalloc( (void**)&resultsd, COMPLEX_ARRAY_SIZE );
    cudaMemcpy( datad, data, COMPLEX_ARRAY_SIZE, cudaMemcpyHostToDevice );
    cudaMemcpy( resultsd, results, COMPLEX_ARRAY_SIZE, cudaMemcpyHostToDevice );

    dim3 dimGrid( 1, 1 );
    dim3 dimBlock( N, 1 );

    // Invoke the kernel
    fft<<<dimGrid, dimBlock>>>(datad, resultsd);

    cudaMemcpy(results, resultsd, COMPLEX_ARRAY_SIZE, cudaMemcpyDeviceToHost);
    cudaFree( datad );
    cudaFree( resultsd );


    // Output the results to fft_output.txt
    outfile << "TOTAL PROCESSED SAMPLES: %i\n";
    outfile << "================================\n";
    // Print X, the results array
    for (int i = 0; i < N; i++)
    {
        outfile << results[i] << '\n';
        outfile << "================================\n";
    }
    
    // Close output file
    outfile.close();
    return 0;
}

__global__ void fft(thrust::complex<float> *datad, thrust::complex<float> *resultsd)
{
    int i = threadIdx.x;
    resultsd[i] = datad[i];
}