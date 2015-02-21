#ifndef BORUVKA_GEN_GPU_HEADER
#define BORUVKA_GEN_GPU_HEADER


#include "common.h"
#include "moderngpu.cuh"
#include "CSR_Graph.cuh"

static unsigned CudaTest(const char *msg)
{
	cudaError_t e;

	//cudaThreadSynchronize();
	cudaDeviceSynchronize();

	if(cudaSuccess != (e = cudaGetLastError()))
	{
		fprintf(stderr, "------======------\n");
		fprintf(stderr, "%s: %d\n", msg, e);
		fprintf(stderr, "%s\n", cudaGetErrorString(e));
		fprintf(stderr, "------======------\n");
		getchar();
		exit(-1);
		return 1;
	}

	return 0;
}

static unsigned int compute_n_blocks(unsigned int problem_size, unsigned int block_size){
	return (problem_size + block_size - 1) / block_size;
}

#endif
