#include "stdio.h"
#include <math.h>
#include <stdlib.h>
static long num_steps = 10000000;
double step;

int main(){
    double pi,sum;
    int steps = num_steps;
    sum=0.0;
    unsigned int seed = 1;
    for (int i=0;i<steps;i++) {
        double val=sqrt((double) pow((double) rand_r(&seed)/ RAND_MAX,2.0)+ (double) pow((double) rand_r(&seed)/ RAND_MAX,2.0));
        if (val<=1.0)
                sum= sum+1.0;
        }


    pi = 4.0*sum/(double) num_steps;
    printf("%f\n\n",pi);
}
