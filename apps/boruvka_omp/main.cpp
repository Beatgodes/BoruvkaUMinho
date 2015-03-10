#include "BoruvkaUMinho_OMP.hpp"

int main(int argc, char *argv[]){
	if(argc != 3)
	{
		printf("Wrong nr of args\n");
		printf("Usage: ./%s <file> <n_threads>\n", argv[0]);
		return 1;
	}

	CSR_Graph *g = new CSR_Graph(argv[1]);
	unsigned int *selected_edges = BoruvkaUMinho_OMP(g, atoi(argv[2]));

	long unsigned int total_weight = 0;

	for(unsigned i = 1; i <= g->nedges ; i++)
	{
		if(selected_edges[i] == 1) total_weight += g->edgessrcwt[i];
	}

	printf("total weight: %lu \n", total_weight*2);
	
	free(selected_edges);
	return 0;
}
