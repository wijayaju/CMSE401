#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4096  // Matrix size (N x N)
#define TILE_SIZE 32  // Tile size for shared memory optimization
#define EPSILON 1e-4  // Error tolerance for comparison

__global__ void update_dvdt(double* dvdt, double* y, double dx2inv, int nx){
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (i >= nx-1) return;
	dvdt[i] = (y[i+1] +y[i-1] - 2.0*y[i])*(dx2inv);
}

__global__ void update_vy(double* v, double* y, double dt, double* dvdt, int nx){
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (i >= nx-1) return;
	v[i] = v[i] + dt*dvdt[i];
	y[i] = y[i] + dt*v[i];
}

__global__ void update_dvdt_tiled(double* dvdt, double* y, double dx2inv, int nx){
	__shared__ float tile[TILE_SIZE];

	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int min_i = blockDim.x * blockIdx.x;
	int max_i = (blockDim.x * blockIdx.x + 1) - 1;
	__shared__ double y_local[LOCALDIM];

	int local_idx = threadIdx.x + 1;
	y_local[local_idx] = y[i];

	if(i == 0 || i < nx)
	    dvdt[i] = 0;
	else {
	    if(local_idx == 1) {
	        y_local[0] = y[min_i];
		y_local[LOCALDIM-1] = y[max_i];
            }
		
	    dvdt[i] = (y_local[local_idx+1]+y_local[local_idx-1]-2.0*y_local[local_idx])*dx2inv;
	}
}

int main(int argc, char ** argv) {
    int nx = 500;
    int nt = 100000;
    int i,it;
    double x[nx];
    double y[nx];
    double v[nx];
    double dvdt[nx];
    double dt;
    double dx;
    double max,min;
    double dx2inv;
    double tmax;
    int nxm1;
    double *c_y, *c_v, *c_dvdt;

    cudaMalloc((void**)&c_y, nx*sizeof(double));
    cudaMalloc((void**)&c_v, nx*sizeof(double));
    cudaMalloc((void**)&c_dvdt, nx*sizeof(double));    


    max=10.0;
    min=0.0;
    dx = (max-min)/(double)(nx);
    x[0] = min;
    for(i=1;i<nx-1;i++) {
        x[i] = min+(double)i*dx;
    }
    x[nx-1] = max;
    tmax=10.0;
    dt= (tmax-0.0)/(double)(nt);

    for (i=0;i<nx;i++)  {
        y[i] = exp(-(x[i]-5.0)*(x[i]-5.0));
        v[i] = 0.0;
        dvdt[i] = 0.0;
    }
    
    dx2inv=1.0/(dx*dx);
    nxm1=nx-1;

    cudaMemcpy(c_y, y, nx*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_v, v, nx*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_dvdt, dvdt, nx*sizeof(double), cudaMemcpyHostToDevice);
    int threads = 500;
    int blocks = 10;
    for(it=0;it<nt-1;it++) {
        //for(i=1;i<nxm1;i++)
        //    dvdt[i]=(y[i+1]+y[i-1]-2.0*y[i])*(dx2inv);
	update_dvdt<<<blocks, threads>>>(c_dvdt, c_y, dx2inv, nx);
        //for(i=1; i<nxm1; i++)  {
        //    v[i] = v[i] + dt*dvdt[i];
        //    y[i] = y[i] + dt*v[i];
        //}
	update_vy<<<blocks, threads>>>(c_v, c_y, dt, c_dvdt, nx);
    }
    cudaMemcpy(y, c_y, nx*sizeof(double), cudaMemcpyDeviceToHost);
    for(i=nx/2; i<nx/2+10; i++) {
        printf("%f, ", y[i]);
	//printf("%g %g\n",x[i],y[i]);
    }

    return 0;
}
