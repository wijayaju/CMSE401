#include <stdio.h>
#include <mpi.h>

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

int main(int argc, char **argv) {
    int rank, size, err_code;
    int msg1;
    int msg2;
    int msg3;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    if (rank == 0) {
        msg1=rank;
        err_code=MPI_Send(&msg1, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        checkError(err_code, rank);
	err_code=MPI_Recv(&msg3, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, &status);
        checkError(err_code, rank);
    } else if (rank == 1) {
        msg2=rank;
	
        err_code=MPI_Send(&msg2, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
        checkError(err_code, rank);
        err_code=MPI_Recv(&msg1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        checkError(err_code, rank);
    } else if (rank == 2){
	msg3=rank;

	err_code=MPI_Send(&msg3, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	checkError(err_code, rank);
	err_code=MPI_Recv(&msg2, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
	checkError(err_code, rank);
    }
    if (rank == 0){
        printf("I am rank %d and received: %d \n",rank, msg3);
    } else if (rank == 1){
        printf("I am rank %d and received: %d \n",rank, msg1);
    } else if (rank == 2){
	printf("I am rank %d and received: %d \n",rank, msg2);
    }
    MPI_Finalize();
    return 0;
}
