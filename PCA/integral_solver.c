#include <iostream>
#include <omp.h>
static long num_steps  = 1000000000;
double step;
int main()
{
    double pi=0.0;
    step = 1.0/(double) num_steps;
    omp_set_num_threads(10);
    double omp_get_wtime;
    #pragma omp parallel for
    for (int i=0;i<num_steps;i++) 
    {
	double x, sum=0.0;
        x = (i + 0.5) * step;
    #pragma omp barrier
	sum = sum+4.0/(1.0+x*x);
	pi = step * sum;
    }
    omp_get_wtime;
    std::cout << pi << std::endl;
}
