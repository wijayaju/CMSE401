#include <omp.h>
#include <vector>
#include <iostream>

// uses unspecified name for critical region.
void foo()
{
    #pragma omp parallel
    #pragma omp critical
    {
        auto id = omp_get_thread_num();
        std::cout << "o,hai there, i'm " << id << '\n';
    }

}

// goes with foo above.
// uses an unspecified name for critical region.
void deadlockA()
{
    int sum = 0;

    #pragma omp parallel for
    for (int ii=0; ii<100; ++ii)
    {
        #pragma omp critical
        {
            sum += ii;
            foo();
        }
    }
}


// slightly different than A.  causes deadlock
// even when using names for the critical sections
void deadlockB()
{
    #pragma omp parallel
    {
        #pragma omp critical(A)
        {
		std::cout << "whassup\n";
        }
        #pragma omp critical(B)
        {
		std::cout << "ahoy\n";
        }
    }

}



int main()
{
    deadlockB();
    return 0;
}
