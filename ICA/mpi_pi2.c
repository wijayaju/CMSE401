#include <mpi.h>
#include <stdio.h>
static long num_steps = 100000; double step;

void checkError(int comm, int rank)
{
    if (comm != MPI_SUCCESS)
    {
    	char error_string[MPI_MAX_ERROR_STRING];
	int length_of_error_string;
	MPI_Error_string(comm, error_string, &length_of_error_string);
	printf("Rank %d: Caught MPI error: %s\n", rank, error_string);
    }
}

int main(int argc, char** argv)
{ 
    int i, nthreads; double pi, sum;
    step = 1.0/(double) num_steps;
    int rank, size, err_code;

    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);  // Set error handler
    err_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Rank (ID) of this process
    checkError(err_code, rank);
    err_code = MPI_Comm_size(MPI_COMM_WORLD, &size); // Total size of MPI job
    checkError(err_code, rank);
    MPI_Status status;

     {
         int i, id,nthrds;
         double x;
         id = rank;
         nthrds = size;
         nthreads = nthrds;
         for (i=id, sum=0.0;i< num_steps; i=i+nthrds) {
             x = (i+0.5)*step;
             sum += 4.0/(1.0+x*x);
         }
     }
    
   
    if (rank == 0) {    
        double procsum;
        pi = sum * step;
        for(int proc=1;proc<nthreads;proc++)
        {
            /* recv sums from all other processors */
            err_code = MPI_Recv(&procsum,1,MPI_DOUBLE,proc,1,MPI_COMM_WORLD, &status);
            checkError(err_code, rank);
	    pi += procsum * step;
        }
        printf("Pi = %f\n",pi);
    } else {
        /*Send rank 0 my sum*/
        err_code = MPI_Send(&sum,1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
	checkError(err_code, rank);
    }
    

    MPI_Finalize();
}
