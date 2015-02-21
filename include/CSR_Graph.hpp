//
//  AL_Graph.h
//  MST
//
//  Compressed sparse row, used for normal graph representation
//  Created by Cristiano Sousa on 17/11/13.
//  Copyright (c) 2013 Cristiano Sousa. All rights reserved.
//

#ifndef CSR_GRAPH_HEADER
#define CSR_GRAPH_HEADER

#include "common.h"
#include <stdlib.h>
#include <stack>
//#include <tuple>
#include <map>
#include <climits>

class CSR_Graph{

public:
	
	unsigned int nnodes;
	unsigned int nedges;
	
	// has nnodes positions
	unsigned int *psrc;         // maps each node to the first edge id
	unsigned int *outdegree;    // maps each node to the number of outgoing edges
	
	// first position of both these arrays act as null
	// has nedges + 1 positions
	unsigned int *edgessrcdst;  // maps each edge to its destination
	unsigned int *edgessrcwt;   // maps each edge to its weight
	
	unsigned int getOutDegree(unsigned int src);
	unsigned int getDestination(unsigned int src, unsigned int nthedge);
	unsigned int getWeight(unsigned int src, unsigned int nthedge);
	unsigned int getFirstEdge(unsigned int src);
	unsigned int findDestination(unsigned int src, unsigned int dst, unsigned wt);
	
	int readFromFile(char *filename);
	int writeToFile(char *filename);
	void allocate();
	void allocate_nodes();
	void allocate_edges();

	void deallocate();
	
	unsigned find_duplicates();
	bool check();
	bool directed();
	void toString();
	long unsigned int getTotalWeight();
	bool equals(CSR_Graph *g);
	bool connected();
	void traverse_graph(bool *visited, unsigned int node);
	bool exists_path(unsigned int src, unsigned int dst);
	
	
	CSR_Graph(char *filename);
	CSR_Graph();
	CSR_Graph(unsigned int t_nnodes, unsigned int t_nedges);

	~CSR_Graph();
	
};

#endif
