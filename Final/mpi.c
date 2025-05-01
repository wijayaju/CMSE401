#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NX 10
#define NSTEPS 1000
#define ALPHA 1.0

double dx = 1.0 / (NX - 1);
double dt;

void save_frame(double u[NX], int step) {
    char filename[64];
    snprintf(filename, sizeof(filename), "heat_step_%03d.dat", step);
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < NX; i++) {
            fprintf(f, "%f ", u[i]);
    }
    fclose(f);
}


int main(int argc, char *argv[]) {
    double u[NX], unew[NX];
    double dudx2;
    int rank, size;
    MPI_Status status[4];
    MPI_Request reqs[4];
    dt = 0.25 * dx*dx / ALPHA;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    //Initialize heatmap
    for (int i = 0; i < NX; i++) {
            double x = i * dx;
            u[i] = exp(-100 * ((x - 0.5)*(x - 0.5)));
            unew[i]=0.0;
    }


    //set initial local values
    double left, center, right;
    double localnew;
    if (rank>0 && rank < NX-1){
        left = u[rank-1];
        center = u[rank];
        right = u[rank+1];
    }
    localnew=unew[rank];

    if (rank==0){
        left = u[NX-1];
        center = u[rank];
        right = u[rank+1];
    }
    if (rank==NX-1){
        left = u[rank-1];
        center = u[rank];
        right = u[0];
    }



    //time loop running NSTEPS timesteps
    for (int n = 0; n < NSTEPS; n++) {
        //heatmap timestep
        // message passing for all non-boundary ranks
        if (rank >0 && rank < NX-1){
            MPI_Isend(&center, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(&center, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[1]);
            MPI_Irecv(&left, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(&right, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[3]); 
        }
        //message passing for rank 0
        if (rank==0){
            //Place send/receive calls here
            MPI_Irecv(&left, 1, MPI_DOUBLE, NX-1, 0, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(&right, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[3]);
            MPI_Isend(&center, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(&center, 1, MPI_DOUBLE, NX-1, 0, MPI_COMM_WORLD, &reqs[1]);

        }
        //message passing for rank NX-1
        if (rank==NX-1){
            //Place send/receive calls here
            MPI_Isend(&center, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(&center, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[1]);
            MPI_Irecv(&left, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(&right, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[3]); 
            
        }

        //wait for all non-blocking message passing to finish
        MPI_Waitall(4, reqs, status);

        dudx2 = (right - 2*center + left) / (dx*dx);
        localnew = center + ALPHA * dt * (dudx2);
        
        //replace center value
        center = localnew;
    }
    
    //For Question 4.4
    MPI_Gather(&center, 1, MPI_DOUBLE, u, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank==0){
        //output final heatmap
        save_frame(u,NSTEPS);
    }
    MPI_Finalize();
    return 0;
}
