#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 1000
#define NY 1000
#define NSTEPS 100
#define ALPHA 1.0

double dx = 1.0 / (NX - 1);
double dy = 1.0 / (NY - 1);
double dt;

void save_frame(double u[NX][NY], int step) {
    char filename[64];
    snprintf(filename, sizeof(filename), "heat_step_%03d.dat", step);
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(f, "%f ", u[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}


int main() {
    double u[NX][NY], unew[NX][NY];
    dt = 0.25 * fmin(dx*dx, dy*dy) / ALPHA;


    //Initialize heatmap
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double x = i * dx;
            double y = j * dy;
            u[i][j] = exp(-100 * ((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5)));
        }
    }

    //time loop running NSTEPS timesteps
    for (int n = 0; n < NSTEPS; n++) {
        //heatmap timestep
        for (int i = 1; i < NX - 1; i++) {
            for (int j = 1; j < NY - 1; j++) {
                double dudx2 = (u[i+1][j] - 2*u[i][j] + u[i-1][j]) / (dx*dx);
                double dudy2 = (u[i][j+1] - 2*u[i][j] + u[i][j-1]) / (dy*dy);
                unew[i][j] = u[i][j] + ALPHA * dt * (dudx2 + dudy2);
            }
        }
        

        //boundary condition
        for (int i = 0; i < NX; i++) {
            u[i][0] = 0.0;
            u[i][NY - 1] = 0.0;
        }
        for (int j = 0; j < NY; j++) {
            u[0][j] = 0.0;
            u[NX - 1][j] = 0.0;
        }

        //replace heatmap
        for (int i = 0; i < NX; i++)
            for (int j = 0; j < NY; j++)
                u[i][j] = unew[i][j];
        
        //output heatmap if interested in verifying result (Note: This will slow down code substantially)
//        save_frame(u, n);
    }

    return 0;
}
