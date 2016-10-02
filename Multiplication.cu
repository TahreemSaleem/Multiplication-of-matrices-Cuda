


#include <stdio.h>
#include <math.h>
#include <iostream>
#include <time.h>

using namespace std;



__global__ void multSquareMatrix(int *A, int *B, int *result, int n)
{
	int k, sum = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	for (k = 0; k < n; k++) {
		sum += A[row * n + k] * B[k * n + col];
		result[row * n + col] = sum;
	}
}


#define N 32

void initMat(int* mat) {
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			mat[i * N + j] = 1 + i + j;
			//	printf("%d \t", mat[i * N + j]);
		}
		//printf("\n");
	}
}

int main() {


	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	int xBlock = props.maxThreadsDim[0];
	int yBlock = props.maxThreadsDim[1];


	if (xBlock > N) {
		xBlock = N;
	}
	if (yBlock > N) {
		yBlock = N;
	}

	int xGrid = props.maxGridSize[0];
	int yGrid = props.maxGridSize[1];

	if (xGrid > ceil(1.0 * N / xBlock)) {
		xGrid = ceil(1.0 * N / xBlock);
	}
	if (yGrid > ceil(1.0 * N / yBlock)) {
		yGrid = ceil(1.0 * N / yBlock);
	}


	dim3 dimBlock(xBlock, yBlock);
	dim3 dimGrid(xGrid, yGrid);
	//start = clock();
	int *arr1_h = (int*)malloc(sizeof(int) * N * N);
	int *arr2_h = (int*)malloc(sizeof(int) * N * N);


	initMat(arr1_h);
	initMat(arr2_h);


	int *arr1_d, *arr2_d, *result_d;
	cudaMalloc(&arr1_d, sizeof(int) * N * N);
	cudaMalloc(&arr2_d, sizeof(int) * N * N);
	cudaMalloc(&result_d, sizeof(int) * N * N);

	cudaMemcpy(arr1_d, arr1_h, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(arr2_d, arr2_h, sizeof(int) * N * N, cudaMemcpyHostToDevice);

	multSquareMatrix << <dimBlock, dimGrid >> >(arr1_d, arr2_d, result_d, N);

	int *result_h = (int*)malloc(sizeof(int) * N * N);
	cudaMemcpy(result_h, result_d, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

	cudaFree(result_d);
	cudaFree(arr1_d);
	cudaFree(arr2_d);
	free(arr1_h);
	free(arr2_h);
	free(result_h);

	return 0;
}