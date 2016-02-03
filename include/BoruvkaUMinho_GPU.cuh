#ifndef BORUVKA_GEN_GPU_HEADER
#define BORUVKA_GEN_GPU_HEADER


#include "common.h"
#include "moderngpu.cuh"
#include "CSR_Graph.cuh"

MGPU_MEM(unsigned int) BoruvkaUMinho_GPU(CSR_Graph *g, unsigned n_blocks);


#endif
