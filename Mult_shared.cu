#include "cuda.h"
#include "stdio.h"
#define BLOCK_X 32
#define BLOCK_Y  32

__global__ void mult_global (int *A, int *B, int *result, int n) 
{
	int k, sum = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(col < n && row  < n) 
	{
		for (k = 0; k < n; k++)
		{
 			sum += A[row * n + k] * B[k * n + col];
	 		result[row * n + col] = sum;
 		}
 	}
}
__global__ void mult_shared( int *A, int *B, int *result, int n) 
{	int k;
	int kk;
 	const int bx = BLOCK_X, by = BLOCK_Y;
	const int col = blockIdx.x*bx + threadIdx.x;
 	const int row = blockIdx.y*by + threadIdx.y;
 	
	__shared__ int a[BLOCK_X][BLOCK_Y] , b[BLOCK_X][BLOCK_Y];
	if ((col < n) && (row < n))
	{
	 	int c = 0;
	 	for (k=0; k < n; k++)
	 	{
	 		a[threadIdx.x][threadIdx.y] = A[ col * n + k*by + threadIdx.y];
	 		b[threadIdx.y][threadIdx.x] = B[ row + n * (k*bx+threadIdx.x)];
	 		__syncthreads(); // Synchronizes all threads in a block
	 		for (kk=0; kk< bx; kk++)
	 			c += a[kk][threadIdx.x]*b[kk][threadIdx.y];
	 		__syncthreads(); // Avoids memory hazards
	 	}
	 	result[col*n+row] = c;
	}

}
int main() {
	
	const int N = 32;

	int *mat1_h = (int *)malloc(sizeof(int) * N * N);
	int *mat2_h = (int *)malloc(sizeof(int) * N * N);

	int *mat1_d, *mat2_d, *result_d;
	cudaMalloc(&mat1_d, sizeof(int) * N * N);
	cudaMalloc(&mat2_d, sizeof(int) * N * N);
	cudaMalloc(&result_d, sizeof(int) * N * N);

	cudaMemcpy(mat1_d, mat1_h, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(mat2_d, mat2_h, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	dim3 dimBlock(256, 256);
	dim3 dimGrid(N/256, N/256);

	mult_shared<<<dimGrid, dimBlock>>>(mat1_d, mat2_d, result_d, N);

	int *result_h = (int *)malloc(sizeof(int) * N);
	cudaMemcpy(result_h, result_d, sizeof(int) * N, cudaMemcpyDeviceToHost);

	//print results

	cudaFree(result_d);
	cudaFree(mat1_d);
	cudaFree(mat2_d);
	free(mat1_h);
	free(mat2_h);
	free(result_h);
	
}
