#include "BoruvkaUMinho_GPU.cuh"

__global__
void load_weights(CSR_Graph g, unsigned int *selected_edges, unsigned int *vertex_minweight){
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= g.nedges + 1) return;

	if(selected_edges[id] == 1)
	{
		vertex_minweight[id] = g.edgessrcwt[id];
	}
}

int main(int argc, char *argv[]){
	if(argc != 2)
	{
		printf("Wrong nr of args\n");
		return 1;
	}

	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0, NULL, true);

	CSR_Graph *g = new CSR_Graph(argv[1]);
	unsigned block_size = atoi(argv[2]);

	MGPU_MEM(unsigned int) selected_edges = BoruvkaUMinho_GPU(g, block_size);

	long unsigned int total_weight = 0;
	MGPU_MEM(unsigned int) vertex_minweight = context->Fill<unsigned int>(g->nedges+1, 0);

	load_weights<<<compute_n_blocks(g->nedges + 1, block_size), block_size>>>(*d_graph[0], selected_edges->get(), vertex_minweight->get());	
	mgpu::Reduce(vertex_minweight->get(), g->nedges + 1, (long unsigned int)0, mgpu::plus<long unsigned int>(), (long unsigned int*)0, &total_weight, *context);
	CudaTest(const_cast<char*>("mgpu::Reduce failed"));

	unsigned int mst_edges = 0;
	mgpu::Reduce(selected_edges->get(), g->nedges + 1, (unsigned int)0, mgpu::plus<unsigned int>(), (unsigned int*)0, &mst_edges, *context);
	CudaTest(const_cast<char*>("mgpu::Reduce 2 failed"));

	printf("total mst weight %lu (not counting mirrored edges (/2): %lu) and %u edges\n", total_weight*2, total_weight, mst_edges-1);
	

	selected_edges->Free();
	vertex_minweight->Free();

	return 0;
}
