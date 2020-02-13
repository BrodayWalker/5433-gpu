#include<iostream>
#include<fstream>

using namespace std;

const int N = 4096;
const int BLOCKSIZE = 1024;

__global__
void add_me(int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
 	c[i] = a[i] + b[i];
}

int main()
{
    ofstream outfile;
    outfile.open("output.txt");

	int a[N] = {0};
 	int b[N];
    int c[N] = {0};
    int sum = 0;

    // Load b with 1s
    for (int i = 0; i < N; i++)
        b[i] = 1;

 	int *ad;
 	int *bd;
    int *cd;
 	const int isize = N*sizeof(int);

 	cudaMalloc( (void**)&ad, isize );
 	cudaMalloc( (void**)&bd, isize );
    cudaMalloc( (void**)&cd, isize );
 	cudaMemcpy( ad, a, isize, cudaMemcpyHostToDevice );
 	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );
    cudaMemcpy( cd, c, isize, cudaMemcpyHostToDevice );

 	dim3 dimGrid( 4, 1 ); 	
	dim3 dimBlock( BLOCKSIZE, 1 );

 	add_me<<<dimGrid, dimBlock>>>(ad, bd, cd);

 	cudaMemcpy( c, cd, isize, cudaMemcpyDeviceToHost );
 	cudaFree( ad );
	cudaFree( bd );
    cudaFree( cd );

    for (int i = 0; i < N; i++)
        sum += c[i];

    cout << "The sum is: " << sum << '\n';
        

 	return EXIT_SUCCESS;
}
