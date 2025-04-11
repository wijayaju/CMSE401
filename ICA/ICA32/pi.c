#include <time.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#define sqr(x)	((x)*(x))
long random(void);

double dboard(int darts)
     {
     double x_coord,       /* x coordinate, between -1 and 1  */
            y_coord,       /* y coordinate, between -1 and 1  */
            pi,            /* pi  */
            r;             /* random number between 0 and 1  */
     int score,            /* number of darts that hit circle */
         n;
     long rd;
     unsigned long cconst; /* used to convert integer random number */
                           /* between 0 and 2^31 to double random number */
                           /* between 0 and 1  */

     cconst = 2 << (31 - 1); 
     cconst = RAND_MAX;
     score = 0;
	
     /* "throw darts at board" */
     for (n = 1; n <= darts; n++)  {
          /* generate random numbers for x and y coordinates */
          rd = random();
          //printf("Rand - %ld\t",rd);
          r = (double)rd/cconst;
          //printf("%10.8f\n",r);
          x_coord = (2.0 * r) - 1.0;
          r = (double)random()/cconst;
          y_coord = (2.0 * r) - 1.0;

          /* if dart lands in circle, increment score */
          if ((sqr(x_coord) + sqr(y_coord)) <= 1.0)
               score++;
      }

     /* calculate pi */
     pi = 4.0 * (double)score/(double)darts;
     return(pi);
}
int main(int argc, char ** argv) {
 
         /** Inicialize MPI **/
    	double pi, avgPi;
	int rank, size;
	srand(time(NULL));
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);	
     	/* "throw darts at board" */
  	pi = dboard(1000000);
	MPI_Reduce(&pi, &avgPi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
	if (rank==0){
		avgPi /= (double)size;

  		printf("%0.16f\n",avgPi);
	} 
	MPI_Finalize();
}
