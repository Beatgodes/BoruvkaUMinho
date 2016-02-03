#include "CSR_Graph.cuh"

CSR_Graph::~CSR_Graph(){
	//deallocate();
}

/**
 * Creates a new CSR Graph filled with the contents of the file
 *
 **/
CSR_Graph::CSR_Graph(char* filename){
	readFromFile(filename);
}

/**
 * Allocates an empty CSR Graph with a given size
 *
 **/
CSR_Graph::CSR_Graph(unsigned int t_nnodes, unsigned int t_nedges, unsigned sys){
	nnodes = t_nnodes;
	nedges = t_nedges;
	if(sys == HOST)
	{
		allocate(t_nnodes, t_nedges);
	}
}

CSR_Graph::CSR_Graph(){}

/**
 * Allocates the necessary memory for the CSR Graph
 * Needs to have nnodes and nedges set
 **/
void CSR_Graph::allocate(unsigned int t_nnodes, unsigned int t_nedges){
	cudaMallocHost(&psrc, t_nnodes * sizeof(unsigned int));
	cudaMallocHost(&outdegree, t_nnodes * sizeof(unsigned int));
	cudaMallocHost(&edgessrcdst, (t_nedges + 1) * sizeof(unsigned int));
	cudaMallocHost(&edgessrcwt, (t_nedges + 1) * sizeof(unsigned int));
}


void CSR_Graph::d_allocate_nodes(){
	if (cudaMalloc((void **)&psrc, nnodes * sizeof(unsigned int)) != cudaSuccess) CudaTest("allocating psrc failed");
	if (cudaMalloc((void **)&outdegree, nnodes * sizeof(unsigned int)) != cudaSuccess) CudaTest("allocating outdegree failed");	

	cudaMemset(psrc, 0, nnodes * sizeof(unsigned int)); 
	cudaMemset(outdegree, 0, nnodes * sizeof(unsigned int));
	cudaDeviceSynchronize();

}


void CSR_Graph::d_allocate_edges(){
	if (cudaMalloc((void **)&edgessrcdst, (nedges + 1) * sizeof(unsigned int)) != cudaSuccess)  CudaTest("allocating edgessrcdst failed");
	if (cudaMalloc((void **)&edgessrcwt, (nedges + 1) * sizeof(unsigned int)) != cudaSuccess) CudaTest("allocating edgessrcwt failed");

	cudaMemset(edgessrcdst, 0, (nedges + 1) * sizeof(unsigned int));
	cudaMemset(edgessrcwt, 0, (nedges + 1) * sizeof(unsigned int));
	cudaDeviceSynchronize();

}


void CSR_Graph::d_allocate(){
	d_allocate_nodes();
	d_allocate_edges();
}

/**
 * Frees up memory
 *
 **/
void CSR_Graph::deallocate(){
	free(psrc);
	free(outdegree);
	free(edgessrcdst);
	free(edgessrcwt);
}

void CSR_Graph::d_deallocate(){
	cudaFree(psrc);
	cudaFree(outdegree);
	cudaFree(edgessrcdst);
	cudaFree(edgessrcwt);	
}

void CSR_Graph::copyHostToDevice(CSR_Graph *d_graph){
	cudaStream_t stream[6]; 
	unsigned int i;
	for(i = 0; i < 6; ++i) 
	{
		cudaStreamCreate(&stream[i]);
	}

	d_graph->nnodes = nnodes;
	d_graph->nedges = nedges;
	cudaMemcpyAsync(d_graph->psrc, psrc, nnodes * sizeof(unsigned int), cudaMemcpyHostToDevice, stream[2]);
	cudaMemcpyAsync(d_graph->outdegree, outdegree, nnodes * sizeof(unsigned int), cudaMemcpyHostToDevice), stream[3];	
	cudaMemcpyAsync(d_graph->edgessrcdst, edgessrcdst, (nedges + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream[4]);
	cudaMemcpyAsync(d_graph->edgessrcwt, edgessrcwt, (nedges + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream[5]);

	for(int i = 0; i < 6; ++i)
	{
		cudaStreamDestroy(stream[i]);
	}
}

void CSR_Graph::copyDeviceToHost(CSR_Graph *h_graph){
	cudaMemcpy(h_graph->psrc, psrc, (h_graph->nnodes) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_graph->outdegree, outdegree, (h_graph->nnodes) * sizeof(unsigned int), cudaMemcpyDeviceToHost);	
	cudaMemcpy(h_graph->edgessrcdst, edgessrcdst, ((h_graph->nedges) + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_graph->edgessrcwt, edgessrcwt, ((h_graph->nedges) + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}


int CSR_Graph::writeToFile(char *filename){
	FILE *fp = fopen(filename, "w+");
	if(!fp)
	{
		printf("File %s does not exist!\n", filename);
		return 1;
	}
	
	fprintf(fp, "%d %d\n", nnodes, nedges);
	
	unsigned int i, j;
	
	for(i = 0; i < nnodes; i++)
	{
		for(j = 0; j < getOutDegree(i); j++)
		{
			fprintf(fp, "%d %d %d\n", i, getDestination(i, j), getWeight(i, j));
		}
	}
	
	fclose(fp);
	return 0;
}


/**
 * Reads CSR Graph from file
 *
 **/
int CSR_Graph::readFromFile(char *filename){
	FILE *fp = fopen(filename, "r");
	if(!fp)
	{
		printf("File %s does not exist!\n", filename);
		return 1;
	}
	unsigned t_nnodes, t_nedges;
	fscanf(fp, "%d %d\n", &t_nnodes, &t_nedges);
	
	allocate(t_nnodes, t_nedges);
	nnodes = t_nnodes;
	nedges = t_nedges;
	
	printf("%d %d\n", nnodes, nedges);

	unsigned int i = 0;
	unsigned prev_node = 0;
	unsigned tmp_srcnode = 0;
	
	psrc[0] = 1;
	edgessrcdst[0] = t_nnodes;
	edgessrcwt[0] = 0;
	outdegree[tmp_srcnode] = 0;
	
	//unsigned out_tmp = 0;
	while(fscanf(fp, "%d %d %d\n", &tmp_srcnode, &(edgessrcdst[i + 1]), &(edgessrcwt[i + 1])) != EOF)
	{
		//edgessrcsrc[i + 1] = tmp_srcnode;
		if(prev_node == tmp_srcnode)
		{
			outdegree[tmp_srcnode]++;
			//out_tmp++;
		}
		else
		{
			//outdegree[prev_node] = out_tmp;
			//out_tmp = 1;
			
			outdegree[tmp_srcnode] = 1;
			psrc[tmp_srcnode] = i + 1;
			prev_node = tmp_srcnode;
		}
		
		i++;
	}
	
	fclose(fp);
	
	if(i != nedges)
	{
		printf("Read %d edges but file says it has %d\n", i, nedges);
		return 1;
	}
	
	return 0;
	
}


/**
 * Given a source and destination node, tries to find the corresponding edge id
 *
 **/
unsigned int CSR_Graph::findDestination(unsigned int src, unsigned int dst, unsigned wt){
	unsigned int i;
	unsigned int edge = getFirstEdge(src);
	
	for(i = 0; i < getOutDegree(src); i++, edge++)
	{
		if(edgessrcdst[edge] == dst && edgessrcwt[edge] == wt) return edge;
	}
	
	return 0;
	
}

/****************************************************************
 *********************** TOOLS **********************************
 ****************************************************************/

unsigned CSR_Graph::find_duplicates(){
	unsigned total = 0;
	for(unsigned i = 0; i < nnodes; i++)
	{
		for(unsigned j = 0; j < getOutDegree(i); j++)
		{
			unsigned dst = getDestination(i,j);
			unsigned wt = getWeight(i,j);

			for(unsigned k = j + 1; k < getOutDegree(i); k++)
			{
				if(dst == getDestination(i,k))
				{
					printf("Found duplicate edge from %d to %d with weights %d and %d\n", i, dst, wt, getWeight(i,k));
					total += getWeight(i,k);
				}
			}
		}
	}

	return total;
}


bool CSR_Graph::check(){
	//printf("nodes: %d\n", *nnodes);
	//printf("edges: %d\n", *nedges);
	//printf("expcected edges: %d\n", 2*(*nnodes - 1));
	
	printf("Checking if graph is undirected\n");

	bool check = true;

	if(directed())
	{
		printf("Graph is directed! Abort!\n");
		check = false;
	}
	
	printf("Checking if graph is connected\n");
	if(!connected())
	{
		printf("Graph is not connected! Abort!\n");
		check = false;
	}
	
	return check;
}

__global__
void kernel_directed(CSR_Graph g, bool *directed){
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= g.nnodes) return;

	unsigned edge = g.psrc[id];

	bool found = false;

	for(unsigned i = 0; i < g.outdegree[id]; i++, edge++)
	{
		unsigned dst = g.edgessrcdst[edge];
		unsigned wt = g.edgessrcwt[edge];
		found = false;

		for(unsigned j = 0; j < g.outdegree[dst] && !found; j++)
		{
			if(id == g.getDestination(dst, j) && wt == g.getWeight(dst,j))
			{
				found = true;
			}
		}

		if(found == false) *directed = true;
	}
}


bool CSR_Graph::directed(){
	bool h_changed, *d_changed;
	if(cudaMalloc((void **)&d_changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");
	cudaMemset(d_changed, 0, sizeof(bool)); 
	
	unsigned block_size = 512;
	unsigned tmp_nnodes;
	//cudaMemcpy(&tmp_nnodes, nnodes, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	tmp_nnodes = nnodes;

	kernel_directed<<<compute_n_blocks(tmp_nnodes, block_size), block_size>>>(*this, d_changed);
	CudaTest("kernel_directed failed");

	cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);

	return h_changed;
}

bool CSR_Graph::exists_path(unsigned int src, unsigned int target){
	
	//if(unlikely(src == target)) return true;
	
	unsigned int i;
	std::stack<unsigned int> stack;
	bool found = false;
	bool *visited = (bool*)calloc(nnodes, sizeof(bool));

	stack.push(src);
	
	while(!stack.empty() && !found){
		unsigned curr = stack.top();
		stack.pop();

		unsigned out = outdegree[curr];
		unsigned first_edge = psrc[curr];

		for(i = 0; i < out && !found; i++)
		{
			unsigned dst = edgessrcdst[first_edge++];
			if(dst != nnodes)
			{
				if(unlikely(dst == target)) found = true;
				if(!visited[dst]) 
				{
					visited[dst] = true;
					stack.push(dst);
				}
			}
		}
	}
	
	free(visited);
	return found;
}

__global__
void set_start_vertex(unsigned int *visited, unsigned int vertex){
	visited[vertex] = 1;
}

__global__
void propagate(CSR_Graph g, unsigned int *visited, bool *changed){
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= g.nnodes) return;
	if(!visited[id]) return;

	unsigned edge = g.psrc[id];
	unsigned outdegree = g.outdegree[id];

	bool my_changed = false;
	for(unsigned i = 0; i < outdegree; i++, edge++)
	{
		unsigned dst = g.edgessrcdst[edge];
		if(!visited[dst])
		{
			visited[dst] = 1;
			my_changed = true;
		}
	}

	if(my_changed) *changed = true;

}

__global__
void find_unvisited(CSR_Graph g, unsigned int *visited, bool *changed){
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= g.nnodes) return;
	if(!visited[id])
	{
		*changed = true;
	}
}

bool CSR_Graph::connected(){
	bool h_changed, *d_changed;
	if(cudaMalloc((void **)&d_changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

	unsigned block_size = 512;
	unsigned tmp_nnodes;
	//cudaMemcpy(&tmp_nnodes, nnodes, sizeof(unsigned), cudaMemcpyDeviceToHost);
	tmp_nnodes = nnodes;

	unsigned *d_visited;
	if(cudaMalloc((void **)&d_visited, sizeof(unsigned) * tmp_nnodes) != cudaSuccess) CudaTest("allocating changed failed");
	cudaMemset(d_visited, 0, sizeof(unsigned) * tmp_nnodes);
	set_start_vertex<<<1, 1>>>(d_visited, 0);
	CudaTest("set_start_vertex failed");

	do{
		cudaMemset(d_changed, 0, sizeof(bool)); 
		propagate<<<compute_n_blocks(tmp_nnodes, block_size), block_size>>>(*this, d_visited, d_changed);
		CudaTest("propagate failed");
		cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
	} while (h_changed);

	cudaMemset(d_changed, 0, sizeof(bool)); 
	find_unvisited<<<compute_n_blocks(tmp_nnodes, block_size), block_size>>>(*this, d_visited, d_changed);
	CudaTest("find_unvisited failed");

	cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
	CudaTest("final copy failed");

	return !h_changed;
}


/**
 * Computes the total weight of all the edges in the graph
 *
 **/
long unsigned int CSR_Graph::getTotalWeight(){
	long unsigned int total = 0;
	
	for(unsigned int i = 1; i <= nedges; i++)
	{
		total += edgessrcwt[i];
	}
	
	return total;
	
}


