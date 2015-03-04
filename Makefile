
WARNINGS = -Wall -Wextra
SUPPRESS_WARNINGS = -Wno-long-long -Wno-unused-value -Wno-unused-local-typedefs -Wno-sign-compare -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-unused-function -Wno-reorder -Wno-strict-aliasing
OPT = -O3




NVCC = nvcc 
GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS = $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)
NVCCFLAGS = $(OPT) $(GENCODE_FLAGS) --compiler-options $(WARNINGS),$(SUPPRESS_WARNINGS)
NVCC_INCLUDES = -I/usr/local/cuda/include/ -I/home/cpd22840/moderngpu/include



CC = g++ -fopenmp -ltbb
CFLAGS = $(OPT) $(WARNINGS) $(SUPPRESS_WARNINGS) -std=c++11 
LIBS =  -L/home/cpd22840/lib -L/usr/lib64 -Llib
INCLUDE = -Iinclude/ -I/home/cpd22840/tbb42/include  -I/home/cpd22840/include



#GPU

boruvka_gpu: MST/lib/cu_CSR_Graph.o  MST/boruvka_gpu/main.o mgpucontext.o mgpuutil.o
	$(NVCC) $(NVCCFLAGS) $(NVCC_INCLUDES) $(NVCC_LIBS) $^ -o bin/$@

MST/boruvka_gpu/main.o: MST/boruvka_gpu/main.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_INCLUDES) $(NVCC_LIBS) -c $^ -o $@

MST/lib/cu_CSR_Graph.o: MST/lib/cu_CSR_Graph.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_INCLUDES) $(NVCC_LIBS) -c $^ -o $@

mgpucontext.o: /home/cpd22840/moderngpu/src/mgpucontext.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_INCLUDES) $(NVCC_LIBS) -o $@ -c $<

mgpuutil.o: /home/cpd22840/moderngpu/src/mgpuutil.cpp
	$(NVCC) $(NVCCFLAGS) $(NVCC_INCLUDES) $(NVCC_LIBS) -o $@ -c $<


#OMP
BoruvkaUMinho_OMP: apps/boruvka_omp/main.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) -lBoruvkaUMinho_OMP $^ -o bin/$@

libBoruvkaUMinho_OMP: src/BoruvkaUMinho_OMP.cpp include/BoruvkaUMinho_OMP.hpp
	$(CC) -fPIC -shared src/BoruvkaUMinho_OMP.cpp src/CSR_Graph.cpp -o lib/libBoruvkaUMinho_OMP.so $(CFLAGS) $(INCLUDE) $(LIBS)



%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) -c $^ -o $@
