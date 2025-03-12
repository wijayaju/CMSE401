// compile: nvcc 1d_wave.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}  

__global__ void update_dvdt(double * dvdt, double * y, double dx2inv, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i >= nx-1) return;
    dvdt[i] = (y[i+1] + y[i-1] - 2.0 * y[i]) * (dx2inv);
}

__global__ void update_vy(double * v, double * y, double dt, double * dvdt, int nx)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx-1) return;
    v[i] = v[i] + dt * dvdt[i];
    y[i] = y[i] + dt * v[i];
}

int main(int argc, char ** argv) {
    int nx = 5000;
    int nt = 1000000;
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

    // init cuda version of data
    double *c_y, *c_v, *c_dvdt;
    // allocate space for cuda versions
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
    // copy cpu version to gpu version
    cudaMemcpy(c_y, y, nx*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_v, v, nx*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_dvdt, dvdt, nx*sizeof(double), cudaMemcpyHostToDevice);
    int blocks = 10;
    int threads = 500;
    // do CUDA parallelization
    for(it=0;it<nt-1;it++) {
        //for(i=1;i<nxm1;i++)
        //    dvdt[i]=(y[i+1]+y[i-1]-2.0*y[i])*(dx2inv);
	update_dvdt<<<blocks, threads>>>(c_dvdt, c_y, dx2inv, nx);
	/*
        for(i=1; i<nxm1; i++)  {
            v[i] = v[i] + dt*dvdt[i];
            y[i] = y[i] + dt*v[i];
        }
	*/	
	update_vy<<<blocks, threads>>>(c_v, c_y, dt, c_dvdt, nx);
    }
    // copy back from gpu to cpu
    cudaMemcpy(y, c_y, nx*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, c_v, nx*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dvdt, c_dvdt, nx*sizeof(double), cudaMemcpyDeviceToHost);

    for(i=nx/2-10; i<nx/2+10; i++) {
        printf("%g %g\n",x[i],y[i]);
    }

    return 0;
}
