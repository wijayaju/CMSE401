#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 1000
#define NSTEPS 1000000
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
                double dudx2 = (u[i+1] - 2*u[i] + u[i-1]) / (dx*dx);
                unew[i] = u[i] + ALPHA * dt * (dudx2);
        }
        

        //replace heatmap
        for (int i = 0; i < NX; i++)
            u[i] = unew[i];
        
        //output heatmap if interested in verifying result (feel free to comment out to avoid file clutter)
        //if (n%100==0)
        //    save_frame(u, n);
    }
    //output final heatmap
    save_frame(u,NSTEPS);
    return 0;
}
