#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 10
#define NSTEPS 100
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


int main() {
    double u[NX], unew[NX];
    double dudx2;
    dt = 0.25 * dx*dx / ALPHA;


    //Initialize heatmap
    for (int i = 0; i < NX; i++) {
            double x = i * dx;
            u[i] = exp(-100 * ((x - 0.5)*(x - 0.5)));
            unew[i]=0.0;
    }

    //time loop running NSTEPS timesteps
    for (int n = 0; n < NSTEPS; n++) {
        //heatmap timestep
        for (int i = 1; i < NX - 1; i++) {
                dudx2 = (u[i+1] - 2*u[i] + u[i-1]) / (dx*dx);
                unew[i] = u[i] + ALPHA * dt * (dudx2);
        }
        
        // index 0 update
        dudx2 = (u[1] - 2*u[0] + u[NX-1])/ (dx*dx);
        unew[0] = u[0] + ALPHA * dt * (dudx2);

        // index NX-1 update
        dudx2 = (u[0] - 2*u[NX-1] + u[NX-2])/ (dx*dx);
        unew[NX-1] = u[NX-1] + ALPHA * dt * (dudx2); 

        //replace heatmap
        for (int i = 0; i < NX; i++)
            u[i] = unew[i];
        
        //output heatmap if interested in verifying result (feel free to comment out to avoid file clutter)
        if (n%1000==0)
            save_frame(u, n);
    }
    //output final heatmap
    save_frame(u,NSTEPS);
    return 0;
}
