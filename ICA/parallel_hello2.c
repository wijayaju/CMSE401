#include "omp.h"
#include "stdio.h"
int main()
{
	nthreads = omp_get_num_threads();
	#pragma omp parallel for
	{
		for (int i=0;i< nthreads;i++)
		{
			int ID = omp_get_thread_num();
			printf("hello(%d)",ID);
			printf(" world(%d) \n",ID);
		}
	}
}
