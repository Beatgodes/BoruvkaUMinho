BoruvkaUMinho
=============

High Performance Parallel CPU and GPU Implementations of Borůvka's Algorithm


Requirements
=============

 * NVCC 5.0 (or higher) and NVIDIA GPU with support for compute capability 2.0 or higher
 * GCC 4.8 (or higher)
 * Intel TBB 4.2 (or higher)
 * [ModernGPU](http://nvlabs.github.io/moderngpu/) (or higher)


Usage
=============

See format.edges to check the expected input format. You can also choose your own graph source (e.g.: parse the file yourself, output of your function), as long as you map it to the CSR_Graph structure (more details on this tbc)

Run 

    make BoruvkaUMinho_OMP

Or

    make BoruvkaUMinho_GPU

To compile the test program. Check the source to see how to get it working.

Do not forget to include the lib folder into your library path!

    export LD_LIBRARY_PATH=/[path to repo]/BoruvkaUMinho/lib:$LD_LIBRARY_PATH

Example input files can be found in the inputs folder. Larger input files (and those that were used in our performance analysis) can be downloaded from [here](http://www.alunos.di.uminho.pt/~pg22840/pub/inputs.tar.gz)

    ./BoruvkaUMinho_OMP <filename> <n threads>
    ./BoruvkaUMinho_GPU <filename> <block size>


Further reading
=============

For further information, please refer to the following [file](http://www.alunos.di.uminho.pt/~pg22840/index.html#pubs) 

If you find this work useful, please cite the following publication:

    Cristiano da Silva Sousa, Artur Mariano and Alberto Proença. "A Generic and Highly Efficient Parallel Variant of Boruvka's Algorithm." 23rd Euromicro Internation Conference on Parallel, Distributed, and Network-Based Processing, 2015

Further help
=============

If you have any question, feel free to contact me.
