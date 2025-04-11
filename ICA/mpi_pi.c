#include <mpi.h>
static long num_steps = 100000; double step;
void main ()
{ 
    double pi, sum, total;
    int size, rank;
    MPI_INIT(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int i, id;
    id = rank;
    double x;
    if(rank!=0){
    	for (i=id, sum=0.0;i< num_steps; i=i+size) {
             x = (i+0.5)*step;
             sum += 4.0/(1.0+x*x);
         }
	MPI_Send(&sum, sizeof(double), MPI_DOUBLE,);
    }else{
    	MPI_Recv(&sum, ); // fix, use vector or something
    }
    for(i=0, pi=0.0;i<nthreads;i++)pi += sum[i] * step;
    MPI_Finalize();
    return pi;
}
