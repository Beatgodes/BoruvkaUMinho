#include "CSR_Graph.hpp"

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
CSR_Graph::CSR_Graph(unsigned int t_nnodes, unsigned int t_nedges){
	//allocate(t_nnodes, t_nedges);
	nnodes = t_nnodes;
	nedges = t_nedges;
}

CSR_Graph::CSR_Graph(){
	//cudaMallocHost(&nnodes, sizeof(unsigned int));
	//cudaMallocHost(&nedges, sizeof(unsigned int));	
}

/**
 * Allocates the necessary memory for the CSR Graph
 * Needs to have nnodes and nedges set
 **/
void CSR_Graph::allocate(){
	allocate_nodes();
	allocate_edges();
}

void CSR_Graph::allocate_nodes(){
	psrc = (unsigned int*)calloc(nnodes, sizeof(unsigned int));
	outdegree = (unsigned int*)calloc(nnodes, sizeof(unsigned int));
}

void CSR_Graph::allocate_edges(){
	edgessrcdst = (unsigned int*)calloc((nedges + 1), sizeof(unsigned int));
	edgessrcwt = (unsigned int*)calloc((nedges + 1), sizeof(unsigned int));
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
	
	nnodes = t_nnodes;
	nedges = t_nedges;	
	allocate();

	
	//printf("%d %d\n", nnodes, nedges);

	unsigned int i = 0;
	unsigned prev_node = 0;
	unsigned tmp_srcnode = 0;
	
	psrc[0] = 1;
/*	edgessrcdst[0] = t_nnodes;
	edgessrcwt[0] = 0;
	outdegree[tmp_srcnode] = 0;
	*/

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
					//printf("Found duplicate edge from %d to %d with weights %d and %d\n", i, dst, wt, getWeight(i,k));
					//total += getWeight(i,k);
					total++;
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


bool CSR_Graph::directed(){
	unsigned i = 0, j = 0;
	
	std::map<std::tuple<unsigned, unsigned, unsigned>, unsigned> pending;
	std::map<std::tuple<unsigned, unsigned, unsigned>, unsigned>::iterator find;

	for(i = 0; i < nnodes; i++)
	{
		for(j = 0; j < getOutDegree(i); j++)
		{
			unsigned dst = getDestination(i, j);
			unsigned wt = getWeight(i, j);
			
			find = pending.find(std::tuple<unsigned, unsigned, unsigned>(dst, i, wt));
			
			if(find == pending.end())
			{
				find = pending.find(std::tuple<unsigned, unsigned, unsigned>(i, dst, wt));
				
				if(find != pending.end()) find->second++;
				else pending.insert(std::pair<std::tuple<unsigned, unsigned, unsigned>, unsigned>(std::tuple<unsigned, unsigned, unsigned>(i, dst, wt), 1));
			}
			else
			{
				if(find->second == 1) pending.erase(find);
				else
					{
						find->second--;
						//find = pending.find(std::tuple<unsigned, unsigned, unsigned>(i, dst, wt));
						//find->second--;
					}
			}
		}
	}
	
	/*for(find = pending.begin(); find != pending.end(); find++)
	{
		std::tuple<unsigned, unsigned, unsigned> f = find->first;
		printf("%d %d %d\n",std::get<0>(f), std::get<1>(f), std::get<2>(f));
	}*/
	
	
	return (pending.size() != 0);
	
	
	//return false;
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




void CSR_Graph::traverse_graph(bool *visited, unsigned int node){
	unsigned int i;
	std::stack<unsigned int> stack;
	
	stack.push(node);
	
	while(!stack.empty()){
		unsigned curr = stack.top();
		stack.pop();
		if(visited[curr]) continue;
		
		visited[curr] = true;
		
		for(i = 0; i < getOutDegree(curr); i++)
		{
			unsigned dst = getDestination(curr, i);
			if(!visited[dst]) stack.push(dst);
		}
	}
}



bool CSR_Graph::connected(){
	bool *visited = (bool*)calloc(nnodes, sizeof(bool));
	unsigned int i;
	
	traverse_graph(visited, 0);
	
	for(i = 0; i < nnodes; i++)
	{
		if(!visited[i])
		{
			printf("no edge to %d\n", i);
			return false;
		}
	}
	
	free(visited);
	return true;
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