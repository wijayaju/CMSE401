#include "stdio.h"
#include "stdlib.h"
#include "time.h"

int main(int argc, char *argv[])
{
    int sz_m=2,sz_k=10,sz_n=3;

    srand(time(0));

    // Accept input numbers for array sizes (m,k,n)
    if (argc > 1)
        sz_m = atoi(argv[1]);

    if (argc > 2)
        sz_k = atoi(argv[2]);

    if (argc > 3)
        sz_n = atoi(argv[3]);

    //Allocate space for matrix A
    double * A_vector = (double *) malloc(sz_m*sz_k*sizeof(double));
    for (int i=0; i<sz_m*sz_k; i++)
        A_vector[i] = rand()%10;
    double **A = malloc(sz_m * sizeof(double*));
    for (int r=0; r<sz_m; r++)
        A[r] = &A_vector[r*sz_k];
 
    //Print out Matrix A
    printf("A = \n");
    for (int i=0; i<sz_m;i++) {
        for (int j=0; j<sz_k;j++)
            printf("%f ",A[i][j]);
        printf("\n");
    }
    printf("\n");

    //Allocate space for matrix B    
    double * B_vector = (double *) malloc(sz_k*sz_n*sizeof(double));
    printf("\n");
    for (int i=0; i<sz_k*sz_n; i++)
        B_vector[i] = rand()%10;
    double **B = malloc(sz_k * sizeof(double*));
    for (int r=0; r<sz_k; r++)
        B[r] = &B_vector[r*sz_n];
    
    //Print out matrix B
    printf("B = \n");
    for (int i=0; i<sz_k;i++) {
        for (int j=0; j<sz_n;j++)
            printf("%f ",B[i][j]);
        printf("\n");
    }
    printf("\n");
    
    //Allocate space for matrix C
    double * C_vector = (double *) malloc(sz_m*sz_n*sizeof(double));
    for (int i=0; i<sz_m*sz_n; i++)
        C_vector[i] = 0;
    double **C = malloc(sz_m * sizeof(double*));
    for (int r=0; r<sz_m; r++)
        C[r] = &C_vector[r*sz_n];
    
    printf("multiplying matrices (%dx%d) (%dx%d)\n",sz_m,sz_k,sz_k,sz_n);
    for (int i=0;i<sz_m;i++){
        for(int j=0;j<sz_n;j++){
            C[i][j] = 0;
            for(int k=0;k<sz_k;k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    //Print out matrix C
    printf("C = \n");
    for (int i=0; i<sz_m;i++) {
        for (int j=0; j<sz_n;j++)
            printf("%f ",C[i][j]);
        printf("\n");
    }
    printf("\n");
    
    free(A_vector);
    free(A);
    free(B_vector);
    free(B);
    free(C_vector);
    free(C);    
}
