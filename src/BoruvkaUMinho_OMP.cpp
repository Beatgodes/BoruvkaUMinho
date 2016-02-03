#include "boruvka_generic_cpu.hpp"

#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__); exit(retval); }   

using namespace tbb;

typedef unsigned T;

class Body {
	T sum;
	T* const y;
	const T* const z;
public:
	Body(T y_[], const T z_[]) : sum(0), z(z_), y(y_) {}
	T get_sum() const {return sum;}

	template<typename Tag>
	void operator()(const blocked_range<unsigned int>& r, Tag)
	{
		T temp = sum;
		for(int i = r.begin(); i < r.end(); ++i)
		{
			if(Tag::is_final_scan())
				y[i] = temp;
			temp = temp + z[i];

		}
		sum = temp;
	}

	Body(Body& b, split) : z(b.z), y(b.y), sum(0) {}
	void reverse_join(Body& a) { sum = a.sum + sum;}
	void assign( Body& b ) {sum = b.sum;}
};


inline void swap(unsigned int* a, unsigned int* b){
	int temp = *a;
	*a = *b;
	*b = temp;
}

inline void find_min_per_vertex(CSR_Graph *g, unsigned int *vertex_minedge, unsigned int id){
	unsigned min_edge = 0;
	unsigned min_weight = UINT_MAX;
	unsigned min_dst = g->nnodes;

	unsigned edge = g->psrc[id];
	unsigned last_edge = edge + g->outdegree[id];
	for(; edge < last_edge; edge++)
	{
		unsigned wt = g->edgessrcwt[edge];
		unsigned dst = g->edgessrcdst[edge];
		if(wt < min_weight || (wt == min_weight && dst < min_dst))
		{
			min_weight = wt;
			min_edge = edge;
			min_dst = dst;
		}
	}

	vertex_minedge[id] = min_edge;

	//return g->outdegree[id];
}

inline void remove_duplicates(CSR_Graph *g, unsigned int *vertex_minedge, unsigned int id){
	unsigned int edge = vertex_minedge[id];
	if(unlikely(edge == 0)) return;
	unsigned int dst = g->edgessrcdst[edge];

	unsigned int other_edge = vertex_minedge[dst];
	if(unlikely(other_edge == 0)) return;
	unsigned int other_dst = g->edgessrcdst[other_edge];

	if(id == other_dst && id > dst) // found loop and maintain edge by smaller vertex id
	{
		vertex_minedge[id] = 0;
	}
}

inline void count_new_edges(CSR_Graph *g, CSR_Graph *next, unsigned int *color, unsigned int *new_vertex, unsigned int id){
	unsigned my_color = color[id];
	//unsigned supervertex_id = new_vertex[color[id]];
	unsigned new_edges = 0;
	unsigned edge = g->psrc[id];
	unsigned last_edge = edge + g->outdegree[id];

			// dst ---> weight
	//std::map<unsigned, unsigned> edges;
	//std::map<unsigned, std::pair<unsigned, unsigned> >::iterator find;

	for(; edge < last_edge; edge++)
	{
		//unsigned other_color = new_vertex[color[dst]];
		unsigned other_color = color[g->edgessrcdst[edge]];
		if(my_color != other_color)
		{
			new_edges++;
			/*find = edges->find(other_color);
			if(find == edges->end())
			{
				edges->emplace(other_color, std::pair<unsigned, unsigned>(g->edgessrcwt[edge], edge));
				new_edges++;	
			}
			else
			{
				unsigned wt = g->edgessrcwt[edge];
				if(wt < find->second.first)
				{
					find->second.first = wt;
					find->second.second = edge;
				}
			}*/
		}
	}


	__atomic_fetch_add(&(next->outdegree[new_vertex[my_color]]), new_edges, __ATOMIC_SEQ_CST);

}

inline void insert_new_edges(CSR_Graph *g, CSR_Graph *next, unsigned int *color, unsigned int *new_vertex, unsigned int *topedge_per_vertex, unsigned int *old_map_edges, unsigned int *new_map_edges, unsigned int id){
	unsigned my_color = color[id];
	unsigned supervertex_id = new_vertex[my_color];
	unsigned edge = g->psrc[id];
	unsigned last_edge = edge + g->outdegree[id];

	for(; edge < last_edge; edge++)
	{
		//unsigned other_supervertex = new_vertex[color[g->edgessrcdst[edge]]];
		unsigned other_color = color[g->edgessrcdst[edge]];
		//if(supervertex_id != other_supervertex)

		if(my_color != other_color)
		{
			unsigned top_edge = __atomic_fetch_add(&(topedge_per_vertex[supervertex_id]), 1, __ATOMIC_SEQ_CST);
			next->edgessrcdst[top_edge] = new_vertex[other_color];//other_supervertex;
			next->edgessrcwt[top_edge] = g->edgessrcwt[edge];
			new_map_edges[top_edge] = old_map_edges[edge];
		}
	}
	/*
	for(auto it = edges->begin(); it != edges->end(); it++)
	{
		unsigned top_edge = __atomic_fetch_add(&(topedge_per_vertex[supervertex_id]), 1, __ATOMIC_SEQ_CST);
		next->edgessrcdst[top_edge] = it->first;
		next->edgessrcwt[top_edge] = it->second.first;
		new_map_edges[top_edge] = old_map_edges[it->second.second];		
	}

	edges->clear();	*/
}

unsigned long omp_get_thread_num_wrap(){
	unsigned long tid = omp_get_thread_num();
	return tid;
}


unsigned int* BoruvkaUMinho_OMP(CSR_Graph *g, unsigned size){
	std::vector<CSR_Graph*> it_graph;
	it_graph.push_back(g);

	omp_set_num_threads(size);
	tbb::task_scheduler_init init(size);

	unsigned size = omp_get_max_threads();
	printf("running with %d threads\n", size);

	unsigned int i;
	bool changed;
	unsigned int iteration = 0;
	unsigned int *vertex_minedge = (unsigned int*)calloc(it_graph[0]->nnodes, sizeof(unsigned int));
	unsigned int *color = (unsigned int*)malloc(sizeof(unsigned int) * it_graph[0]->nnodes);
	unsigned int *selected_edges = (unsigned int*)calloc(it_graph[0]->nedges + 1, sizeof(unsigned int));
	unsigned int *new_vertex = (unsigned int*)malloc(sizeof(unsigned int) * it_graph[0]->nnodes);
	unsigned int *topedge_per_vertex = (unsigned int*)malloc(sizeof(unsigned int) * it_graph[0]->nnodes);
	unsigned int *supervertex_flag = (unsigned int*)malloc(sizeof(unsigned int) * it_graph[0]->nnodes);
	//unsigned int *new_map_edges = (unsigned int*)malloc(sizeof(unsigned int) * (it_graph[0]->nedges + 1));
	//unsigned int *map_edges = (unsigned int*)malloc(sizeof(unsigned int) * (it_graph[0]->nedges + 1));
	unsigned int *arr_map_edges[2];
	arr_map_edges[0] = (unsigned int*)malloc(sizeof(unsigned int) * (it_graph[0]->nedges + 1));
	arr_map_edges[1] = (unsigned int*)malloc(sizeof(unsigned int) * (it_graph[0]->nedges + 1));

	std::map<unsigned, unsigned> histogram;

	unsigned curr_map = 0;
	unsigned new_map = 1;

	for(i = 0; i <= it_graph[0]->nedges; i++)
	{
		arr_map_edges[curr_map][i] = i;
		arr_map_edges[new_map][i] = i;
	}

	double time[12];
	for(i = 0; i < 12; i++) time[i] = 0.0f;

	//std::vector<std::map<unsigned, std::pair<unsigned, unsigned> > > map_edges(it_graph[iteration]->nnodes);

	//int retval;
   	//if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT ) ERROR_RETURN(retval);
	//if((retval = ) != PAPI_OK) ERROR_RETURN(retval);
    //PAPI_thread_init(omp_get_thread_num_wrap);

    //long long *values = (long long*)malloc(sizeof(long long) * size * 6);
    //int *EventSet = (int*)malloc(sizeof(int) * size);
    //unsigned *edges = (unsigned*)calloc(size, sizeof(unsigned));

	double start_time = omp_get_wtime();
	while(true)
	{
		printf("graph has %u nodes and %u edges\n", it_graph[iteration]->nnodes, it_graph[iteration]->nedges);
		/*#pragma omp parallel private(retval)
		{
			//int EventSet = PAPI_NULL;
			int tid = omp_get_thread_num();
			EventSet[tid] = PAPI_NULL;
			if((retval = PAPI_create_eventset(&EventSet[tid])) != PAPI_OK) ERROR_RETURN(retval);
			//if((retval = PAPI_add_event(EventSet[tid], PAPI_TOT_INS)) != PAPI_OK) ERROR_RETURN(retval);
			if((retval = PAPI_add_event(EventSet[tid], PAPI_L1_TCM)) != PAPI_OK) ERROR_RETURN(retval);
			if((retval = PAPI_add_event(EventSet[tid], PAPI_L2_TCM)) != PAPI_OK) ERROR_RETURN(retval);
			if((retval = PAPI_add_event(EventSet[tid], PAPI_L3_TCM)) != PAPI_OK) ERROR_RETURN(retval);
			//if((retval = PAPI_add_event(EventSet[tid], PAPI_TOT_CYC)) != PAPI_OK) ERROR_RETURN(retval);
			//if((retval = PAPI_add_event(EventSet[tid], PAPI_L3_TCA)) != PAPI_OK) ERROR_RETURN(retval);
			//if((retval = PAPI_add_event(EventSet[tid], PAPI_STL_ICY)) != PAPI_OK) ERROR_RETURN(retval);
			if((retval = PAPI_start(EventSet[tid])) != PAPI_OK) ERROR_RETURN(retval);

		}*/
		//double ttt_start = omp_get_wtime();		

		double it_start_time = omp_get_wtime();
		#pragma omp parallel private(i)
		{
			#pragma omp for schedule(guided)
			for(i = 0; i < it_graph[iteration]->nnodes; i++)
			{
				find_min_per_vertex(it_graph[iteration], vertex_minedge, i);
			}
		}

		double it_end_time = omp_get_wtime();
		time[0] += it_end_time - it_start_time;


		it_start_time = omp_get_wtime();
		#pragma omp parallel for schedule(guided)
		for(i = 0; i < it_graph[iteration]->nnodes; i++)
		{
			remove_duplicates(it_graph[iteration], vertex_minedge, i);
		}
		it_end_time = omp_get_wtime();
		time[3] += it_end_time - it_start_time;


		it_start_time = omp_get_wtime();
		#pragma omp parallel for schedule(guided)
		for(i = 0; i < it_graph[iteration]->nnodes; i++)
		{
			unsigned edge = vertex_minedge[i];
			if(edge == 0) color[i] = i;
			else color[i] = it_graph[iteration]->edgessrcdst[edge];
		}
		it_end_time = omp_get_wtime();
		time[1] += it_end_time - it_start_time;


		it_start_time = omp_get_wtime();
		do{
			changed = false;

			#pragma omp parallel private(i)
			{
				bool my_changed = false;

				#pragma omp for
				for(i = 0; i < it_graph[iteration]->nnodes; i++)
				{
					unsigned int my_color = color[i];
					unsigned int other_color = color[my_color];

					if(my_color != other_color)
					{
						color[i] = other_color;
						my_changed = true;
					}				
				}

				if(my_changed) changed = true;	
			}

		} while(changed);
		it_end_time = omp_get_wtime();
		time[2] += it_end_time - it_start_time;

		it_start_time = omp_get_wtime();
		#pragma omp parallel for
		for(i = 0; i < it_graph[iteration]->nnodes; i++)
		{
			unsigned int edge = vertex_minedge[i];
			selected_edges[arr_map_edges[curr_map][edge]] = 1;
		}
		it_end_time = omp_get_wtime();
		time[4] += it_end_time - it_start_time;


		it_graph.push_back(new CSR_Graph(0, 0));
		unsigned next_nnodes = 0;





		it_start_time = omp_get_wtime();
		#pragma omp parallel for schedule(guided)
		for(i = 0; i < it_graph[iteration]->nnodes; i++)
		{
			if(unlikely(i == color[i]) && unlikely(it_graph[iteration]->outdegree[i] > 0)) // representative thread
			{	
				supervertex_flag[i] = 1;
			}
			else supervertex_flag[i] = 0;
		}

		Body body(new_vertex, supervertex_flag);
		parallel_scan(blocked_range<unsigned int>(0,it_graph[iteration]->nnodes), body);
		next_nnodes = body.get_sum();
		it_end_time = omp_get_wtime();
		time[5] += it_end_time - it_start_time;


		if(unlikely(next_nnodes == 1)) break;

		it_graph[iteration+1]->nnodes = next_nnodes;
		it_graph[iteration+1]->allocate_nodes();


		it_start_time = omp_get_wtime();
		memset(it_graph[iteration+1]->outdegree, 0, next_nnodes * sizeof(unsigned int));
		it_end_time = omp_get_wtime();
		time[9] += it_end_time - it_start_time;


		it_start_time = omp_get_wtime();
		#pragma omp parallel for schedule(guided)
		for(i = 0; i < it_graph[iteration]->nnodes; i++)
		{
			count_new_edges(it_graph[iteration], it_graph[iteration+1], color, new_vertex, i);
		}
		it_end_time = omp_get_wtime();
		time[6] += it_end_time - it_start_time;

		it_graph[iteration+1]->nedges = 1;

		it_start_time = omp_get_wtime();
		Body body2(it_graph[iteration+1]->psrc, it_graph[iteration+1]->outdegree);
		parallel_scan(blocked_range<unsigned int>(0,it_graph[iteration+1]->nnodes), body2);
		it_graph[iteration+1]->nedges = body2.get_sum();

		#pragma omp parallel for schedule(guided)
		for(i = 0; i < it_graph[iteration+1]->nnodes; i++)
		{
			it_graph[iteration+1]->psrc[i]++;
		}

		it_end_time = omp_get_wtime();
		time[7] += it_end_time - it_start_time;


		it_graph[iteration+1]->allocate_edges();

		it_start_time = omp_get_wtime();		
		memcpy(topedge_per_vertex, it_graph[iteration+1]->psrc, sizeof(unsigned int) * next_nnodes);
		it_end_time = omp_get_wtime();
		time[10] += it_end_time - it_start_time;

		it_start_time = omp_get_wtime();	
		#pragma omp parallel for schedule(guided)
		for(i = 0; i < it_graph[iteration]->nnodes; i++)
		{
			insert_new_edges(it_graph[iteration], it_graph[iteration+1], color, new_vertex, topedge_per_vertex, arr_map_edges[curr_map], arr_map_edges[new_map], i);
		}

		it_end_time = omp_get_wtime();
		time[8] += it_end_time - it_start_time;


		swap(&curr_map, &new_map);
		//double ttt_end = omp_get_wtime();
		
		if(iteration > 0) it_graph[iteration]->deallocate();


		/*if(it_graph[iteration+1]->nnodes < it_graph[iteration]->nnodes / 3.5)
		{
			//size = size / 2;
			if(size > 1) size-=2;
			omp_set_num_threads(size);	
			printf("now running with %u threads\n", size);
		}*/
/*
		memset(values, 0, sizeof(long long) * size * 6);
		#pragma omp parallel private(retval)
		{
			int tid = omp_get_thread_num();
			if((retval = PAPI_stop(EventSet[tid], &values[tid*6])) != PAPI_OK) ERROR_RETURN(retval);
			//if((retval = PAPI_remove_event(EventSet[tid], PAPI_TOT_INS)) != PAPI_OK) ERROR_RETURN(retval);
			if((retval = PAPI_remove_event(EventSet[tid], PAPI_L1_TCM)) != PAPI_OK) ERROR_RETURN(retval);
			if((retval = PAPI_remove_event(EventSet[tid], PAPI_L2_TCM)) != PAPI_OK) ERROR_RETURN(retval);
			if((retval = PAPI_remove_event(EventSet[tid], PAPI_L3_TCM)) != PAPI_OK) ERROR_RETURN(retval);
			//if((retval = PAPI_remove_event(EventSet[tid], PAPI_TOT_CYC)) != PAPI_OK) ERROR_RETURN(retval);
			//if((retval = PAPI_remove_event(EventSet[tid], PAPI_L3_TCA)) != PAPI_OK) ERROR_RETURN(retval);
			//if((retval = PAPI_remove_event(EventSet[tid], PAPI_STL_ICY)) != PAPI_OK) ERROR_RETURN(retval);
			if((retval = PAPI_destroy_eventset(&(EventSet[tid]))) != PAPI_OK) ERROR_RETURN(retval);

		}
*/
		/*long long sum[6];
		sum[0] = 0;
		sum[1] = 0;
		sum[2] = 0;
		//sum[3] = 0;
		//sum[4] = 0;
		//sum[5] = 0;
		for(i = 0; i < size; i++)
		{
			sum[0] += values[i * 6 + 0];
			sum[1] += values[i * 6 + 1];
			sum[2] += values[i * 6 + 2];
			//sum[4] += values[i * 6 + 4];
			//sum[5] += values[i * 6 + 5];
			//printf("%d cyc %lld stall %lld l2tcm %lld l3tcm %lld\n", i, values[i * 6 + 3], values[i * 6 + 5], values[i * 6 + 2], values[i * 6 + 1]);
		}

		printf("i:%u\t%lld\t%lld\t%lld\t%f\n", iteration, sum[0], sum[1], sum[2], ttt_end - ttt_start);
		//printf("i:%u\t%lld\t%lld\t%lld\t%lld\t%lld\t%f\n", iteration, sum[0], sum[1], sum[2], sum[4], sum[5], ttt_end - ttt_start);

*/
		iteration++;
		//printf("took %f\n", ttt_end - ttt_start);

	}
	double end_time = omp_get_wtime();

	printf("%.1f\t ms on find_min_per_vertex\n", time[0]*1000);
	printf("%.1f\t ms on init_color\n", time[1]*1000);
	printf("%.1f\t ms on propagate_color\n", time[2]*1000);
	printf("%.1f\t ms on remove_duplicates\n", time[3]*1000);	
	//printf("%.1f\t ms on tbb::sort\n", time[11]*1000);
	printf("%.1f\t ms on mark_mst_vertices\n", time[4]*1000);
	printf("%.1f\t ms on create_new_vertex_id\n", time[5]*1000);
	printf("%.1f\t ms on memest outdegree\n", time[9]*1000);
	printf("%.1f\t ms on count_new_edges\n", time[6]*1000);
	printf("%.1f\t ms on setup_psrc\n", time[7]*1000);
	printf("%.1f\t ms on copy topedge_per_vertex\n", time[10]*1000);
	printf("%.1f\t ms on insert_new_edges\n", time[8]*1000);
	printf("%.1f\t total time\n", 1000*(end_time - start_time));

	free(vertex_minedge);
	free(color);
	free(topedge_per_vertex);
	free(arr_map_edges[0]);
	free(arr_map_edges[1]);
	free(new_vertex);
	free(supervertex_flag);
	return selected_edges;
}