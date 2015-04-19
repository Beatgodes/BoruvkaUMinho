OPT = -O3

TBB_INCLUDE_PATH = -I/home/cpd22840/tbb42/include
TBB_LIBRARY_PATH = -L/home/cpd22840/lib

MGPU_INCLUDE_PATH= -I/home/cpd22840/moderngpu/include 

NVCC = nvcc 
CC = g++ -fopenmp -ltbb


WARNINGS = -Wall -Wextra
SUPPRESS_WARNINGS = -Wno-long-long -Wno-unused-value -Wno-unused-local-typedefs -Wno-sign-compare -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-unused-function -Wno-reorder -Wno-strict-aliasing

NVCC_WARNINGS = -Wall,-Wextra
NVCC_SUPPRESS_WARNINGS = -Wno-long-long,-Wno-unused-value,-Wno-unused-local-typedefs,-Wno-sign-compare,-Wno-unused-but-set-variable,-Wno-unused-parameter,-Wno-unused-function,-Wno-reorder,-Wno-strict-aliasing






GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS = $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)
NVCCFLAGS = $(OPT) $(GENCODE_FLAGS) --compiler-options $(NVCC_WARNINGS)
NVCC_INCLUDES = -Iinclude/ $(MGPU_INCLUDE_PATH)
NVCC_LIBS = -Llib/

CFLAGS = $(OPT) $(WARNINGS) $(SUPPRESS_WARNINGS) -std=c++11 
LIBS =  -Llib/ $(TBB_LIBRARY_PATH) 
INCLUDE = -Iinclude/ $(TBB_INCLUDE_PATH)



#usage
apps: BoruvkaUMinho_OMP BoruvkaUMinho_GPU

BoruvkaUMinho_GPU: apps/boruvka_gpu/main.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_INCLUDES) $(NVCC_LIBS) -lBoruvkaUMinho_GPU $^ -o bin/$@

BoruvkaUMinho_OMP: apps/boruvka_omp/main.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) -lBoruvkaUMinho_OMP $^ -o bin/$@




# compile libs

libs: libBoruvkaUMinho_OMP libBoruvkaUMinho_GPU

libBoruvkaUMinho_GPU: src/BoruvkaUMinho_GPU.cu include/BoruvkaUMinho_GPU.cuh
	$(NVCC) --compiler-options '-fPIC' -shared -o lib/libBoruvkaUMinho_GPU.so $(NVCCFLAGS) $(NVCC_INCLUDES) $(NVCC_LIBS)  src/BoruvkaUMinho_GPU.cu src/cu_CSR_Graph.cu /home/cpd22840/moderngpu/src/mgpucontext.cu /home/cpd22840/moderngpu/src/mgpuutil.cpp

libBoruvkaUMinho_OMP: src/BoruvkaUMinho_OMP.cpp include/BoruvkaUMinho_OMP.hpp
	$(CC) -fPIC -shared src/BoruvkaUMinho_OMP.cpp src/CSR_Graph.cpp -o lib/libBoruvkaUMinho_OMP.so $(CFLAGS) $(INCLUDE) $(LIBS)


libBoruvkaUMinho_PHI: src/BoruvkaUMinho_OMP.cpp include/BoruvkaUMinho_OMP.hpp
	icpc -tbb -openmp -mmic -fPIC -shared src/BoruvkaUMinho_OMP.cpp src/CSR_Graph.cpp -o lib/libBoruvkaUMinho_PHI.so \
	$(OPT) $(WARNINGS) -std=c++11 \
	-Iinclude/  -I/opt/intel/composer_xe_2013.1.117/compiler/include/mic/ \
	-L/opt/intel/composer_xe_2013.1.117/compiler/lib/mic/ 
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) -c $^ -o $@
