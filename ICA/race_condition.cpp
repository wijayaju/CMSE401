#include <omp.h>

#include <vector>
#include <iostream>

int main()
{
    int counter = 0;
    int size = 1000;
    #pragma omp parallel for reduction(+:counter)
    for (int ii=0; ii<size; ++ii)
    {
        if (ii%2)
            ++counter;
    }


    std::cout << counter << '\n';

    return 0;
}
