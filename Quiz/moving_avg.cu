#include "stdio.h"
#include <math.h>
#include <stdlib.h>
#include <cuda.h>

#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}  

__global__ void compute_avgs(double * s, double * a, int r)
{   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j=0; j<=r;j++){
	a[i]+=s[i+j];
    }
    a[i] /= (double)r + 1.0;
}
static long num_steps = 10000000;

int main(){
    int steps = num_steps;
    unsigned int seed = 1;
    int range=1000;
    double series[num_steps];
    double avg[num_steps-range];
    
    //Initialize CUDA variables and allocate
    double *c_series, *c_avg;
    cudaMalloc((void**)&c_series, num_steps*sizeof(double));
    cudaMalloc((void**)&c_avg, (num_steps-range)*sizeof(double));

    //Initialize values in list
    series[0]=10.0;
    for (int i=1;i<steps;i++) {
        series[i]=series[i-1]+ ((double) rand_r(&seed))/RAND_MAX-0.5;
    }
    for (int i=0; i<steps-range;i++){
        avg[i]=0;
    }


    //Copy to device
    cudaMemcpy(c_series, series, num_steps*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_avg, avg, (num_steps-range)*sizeof(double), cudaMemcpyHostToDevice);
    int threads = 1000;
    int blocks = (num_steps - range) / threads
    //Compute averages with CUDA parallelization
	compute_avgs<<<blocks, threads>>>(c_series, c_avg, range);
        //for (int j=0; j<=range;j++){
        //    avg[i]+=series[i+j];
        //}
        //avg[i]/=(double)range + 1.0;
    }

    //Copy back to host
    cudaMemcpy(series, c_series, num_steps*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(avg, c_avg, (num_steps-range)*sizeof(double), cudaMemcpyDeviceToHost);

    //Print elements for comparison
    printf("%f %f\n\n",series[steps-1],avg[steps-range-1]);
}
