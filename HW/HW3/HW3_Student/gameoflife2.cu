#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include <assert.h>
#include "png_util.h"
#include <cuda.h>
#define MAX_N 20000

#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}  

char plate[2][(MAX_N + 2) * (MAX_N + 2)];
char *c_plate0, *c_plate1;  // initialize plate copy for gpu
int which = 0;
int n;
int live(int index){
    return (plate[which][index - n - 3] 
        + plate[which][index - n - 2]
        + plate[which][index - n - 1]
        + plate[which][index - 1]
        + plate[which][index + 1]
        + plate[which][index + n + 1]
        + plate[which][index + n + 2]
        + plate[which][index + n + 3]);
}

__global__ void iteration_kernel(char * pin, char * pout, int n){  // kernel for updating plate state with gpu

    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;  // helps code not access out of bound indices through the constraint
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if(i <= n && j <= n){  // constraint to ensure code doesn't access out of bound indices
    	int index = i * (n + 2) + j;
        int num = (pin[index - n - 3]
	    + pin[index - n - 2]
	    + pin[index - n - 1]
	    + pin[index - 1]
	    + pin[index + 1]
	    + pin[index + n + 1]
	    + pin[index + n + 2]
	    + pin[index + n + 3]);

    	if(pin[index] == 1){
	    pout[index] = (num == 2 || num == 3) ? 1 : 0;
    	}else{
	    pout[index] = (num == 3);
    	}
    }
}

void iteration(){
    cudaMemcpy(c_plate0, plate[0], (MAX_N + 2) * (MAX_N + 2) * sizeof(char), cudaMemcpyHostToDevice);  // copy start state of plate 0 to gpu copy
    cudaMemcpy(c_plate1, plate[1], (MAX_N + 2) * (MAX_N + 2) * sizeof(char), cudaMemcpyHostToDevice);  // copy start state of plate 1 to gpu copy

    dim3 dimBlock(16, 16, 1);  // size of block
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y, 1);  // size of grid

    if (which == 0) {
        iteration_kernel<<<dimGrid, dimBlock>>>(c_plate0, c_plate1, n);  // call kernel
    } else {
        iteration_kernel<<<dimGrid, dimBlock>>>(c_plate1, c_plate0, n);  // call kernel
    }
    
    which = !which;

    if (which == 0) {
        cudaMemcpy(plate[which], c_plate0, (MAX_N + 2) * (MAX_N + 2) * sizeof(char), cudaMemcpyDeviceToHost);  // copy updated data back to cpu
    } else {
        cudaMemcpy(plate[which], c_plate1, (MAX_N + 2) * (MAX_N + 2) * sizeof(char), cudaMemcpyDeviceToHost);  // copy updated data back to cpu
    }
}

void print_plate(){
    if (n < 60) {
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= n; j++){
                printf("%d", (int) plate[which][i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
	printf("Plate too large to print to screen\n");
    }
    printf("\0");
}

void plate2png(char* filename) {
    unsigned char * img = (unsigned char *) malloc(n*n*sizeof(unsigned char));
    image_size_t sz;
    sz.width = n;
    sz.height = n; 

    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= n; j++){
            int pindex = i * (n + 2) + j;
            int index = (i-1) * (n) + j;
            if (plate[!which][pindex] > 0)
		img[index] = (unsigned char)255; 
            else 
		img[index] = (unsigned char)0;
        }
    }

    printf("Writing file\n");
    write_png_file(filename,img,sz);
    printf("done writing png\n"); 
    free(img);
    printf("done freeing memory\n");
}

int main() {
    int M;
    char line[MAX_N];
    cudaMalloc((void**)&c_plate0, (MAX_N + 2) * (MAX_N + 2) * sizeof(char));  // allocate space for gpu copy of plate 0
    cudaMalloc((void**)&c_plate1, (MAX_N + 2) * (MAX_N + 2) * sizeof(char));  // allocate space for gpu copy of plate 1
    if(scanf("%d %d", &n, &M) == 2){
	if (n > 0) {
            memset(plate[0], 0, sizeof(char) * (n + 2) * (n + 2));
            memset(plate[1], 0, sizeof(char) * (n + 2) * (n + 2));
            for(int i = 1; i <= n; i++){
                scanf("%s", &line);
                for(int j = 0; j < n; j++){
                    plate[0][i * (n + 2) + j + 1] = line[j] - '0';
                }
            }
	} else {
	   n = MAX_N;
	   for(int i = 1; i <= n; i++) 
               for(int j = 0; j < n; j++) 
                   plate[0][i * (n+2) +j + 1] = (char) rand() % 2;
	}

        for(int i = 0; i < M; i++){
            printf("\nIteration %d:\n",i);
	    print_plate();
            iteration();
        }

        printf("\n\nFinal:\n");
	plate2png("plate.png");
	print_plate();
    }

    cudaFree(c_plate0);  // free up gpu memory
    cudaFree(c_plate1);

    return 0;
}

