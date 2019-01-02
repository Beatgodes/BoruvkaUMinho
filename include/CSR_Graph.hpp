//
//  CSR_Graph.hpp
//  MST
//
//  Compressed sparse row, used for normal graph representation
//  Created by Cristiano Sousa on 17/11/13.
//  Copyright (c) 2013 Cristiano Sousa. All rights reserved.
//

#ifndef CSR_GRAPH_HEADER
#define CSR_GRAPH_HEADER

#include "common.h"
#include <stack>
#include <map>
#include <climits>
#include <tuple>

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
	//unsigned int *edgessrcsrc;   // maps each edge to its source
	
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


/**
 * Returns the outdegree of a given edge
 *
 **/
inline unsigned int CSR_Graph::getOutDegree(unsigned int src){
	if(src < nnodes)
	{
		return outdegree[src];
	}
	
	printf("Error: %s(%d): getOutDegree - node %d out of bounds (%d)\n", __FILE__, __LINE__, src, nnodes);
	return 0;
	
}

/**
 * Returns the edge id of the first edge of a given node
 *
 **/
inline unsigned int CSR_Graph::getFirstEdge(unsigned int src){
	if(src < nnodes && getOutDegree(src) > 0)
	{
		return psrc[src];
	}
	
	if(src >= nnodes) printf("Error: %s(%d): getFirstEdge - node %d out of bounds (%d)\n", __FILE__, __LINE__, src, nnodes);
	if(getOutDegree(src) == 0) printf("Error: %s(%d): getFirstEdge - function called for node %d but has no neighbours\n", __FILE__, __LINE__, src);
	
	return 0;
	
	
}

/**
 * Given a node and a edge number, returns the corresponding destination
 *
 **/
inline unsigned int CSR_Graph::getDestination(unsigned int src, unsigned int nthedge){
	if(src < nnodes && nthedge < getOutDegree(src))
	{
		unsigned int edge = getFirstEdge(src);
		
		if(edge && edge + nthedge < nedges + 1)
		{
			return edgessrcdst[edge + nthedge];
		}
		
		if(!edge) printf("Error: %s(%d): getDestination - node %d out of bounds (%d)\n", __FILE__, __LINE__, src, nnodes);
		
	}
	
	if(src >= nnodes) printf("Error: %s(%d): getDestination - node %d out of bounds (%d)\n", __FILE__, __LINE__, src, nnodes);
	if(nthedge >= getOutDegree(src)) printf("Error: %s(%d): getDestination - node %d out of bounds (%d) edge %d(%d)\n", __FILE__, __LINE__, src, nnodes, nthedge, getOutDegree(src));
	
	
	return nnodes;
}

/**
 * Given a node and a edge number, returns the corresponding weight
 *
 **/
inline unsigned int CSR_Graph::getWeight(unsigned int src, unsigned int nthedge){
	if(src < nnodes && nthedge < getOutDegree(src))
	{
		unsigned int edge = getFirstEdge(src);
		
		if(edge && edge + nthedge < nedges +1)
		{
			return edgessrcwt[edge + nthedge];
		}
		
		if(!edge) printf("Error: %s(%d): getWeight - node %d out of bounds (%d)\n", __FILE__, __LINE__, src, nnodes);
		
	}
	else
	{
		printf("Error: %s(%d): getWeight - node %d out of bounds (%d) or nthedge %d out of bounds (%d)\n", __FILE__, __LINE__, src, nnodes, nthedge, getOutDegree(src));
	}
	
	return UINT_MAX;
}

#endif
