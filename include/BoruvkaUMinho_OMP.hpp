#ifndef BORUVKA_GEN_CPU_HEADER
#define BORUVKA_GEN_CPU_HEADER

#include "common.h"
#include <omp.h>



#include "CSR_Graph.hpp"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_sort.h"
#include "tbb/parallel_scan.h"

unsigned int* BoruvkaUMinho_OMP(CSR_Graph *g, unsigned n_threads);

#endif
