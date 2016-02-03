#ifndef GPU_UTILS
#define GPU_UTILS

#define HOST 0
#define DEVICE 1

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

static int ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{ 
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{   -1, -1 }
	};

	int index = 0;
	while(nGpuArchCoresPerSM[index].SM != -1) 
	{
		if(nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}
	printf("MapSMtoCores SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
	return -1;
}

static unsigned int compute_n_blocks(unsigned int problem_size, unsigned int block_size){
	return (problem_size + block_size - 1) / block_size;
}

static int detect_devices(){
	int deviceCount = 0;
	if (cudaSuccess != cudaGetDeviceCount(&deviceCount))
	{
		CudaTest("cudaGetDeviceCount failed");
	}

	if (deviceCount == 0)
	{
		fprintf(stderr, "No CUDA capable devices found.");
		return 1;
	}

	cudaDeviceProp dp;	
	fprintf(stdout, "Found %d devices\n", deviceCount);

	for(int device = 0; device < deviceCount; device++)
	{
		cudaGetDeviceProperties(&dp, device);
		printf("Device %d (%s), compute capability %d.%d, %d SMs and %d cores/SM.\n", 
			device, dp.name, dp.major, dp.minor, dp.multiProcessorCount, ConvertSMVer2Cores(dp.major, dp.minor));
	}

	return 0;
}

#endif