//
// Created by Verma, Navneet on 7/7/23.
//
#include <omp.h>
#include "cstdio"

int main() {

    int nthreads, tid, maxth;

    /* Fork a team of threads giving them their own copies of variables */

    omp_set_num_threads(4);
#pragma omp parallel private(tid)
    {
        /* Obtain thread number */
        //omp_set_num_threads(4);
        tid = omp_get_thread_num();
        maxth=omp_get_max_threads();
        printf("Hello World from thread = %d\n", tid);

        /* Only master thread does this */

        // #pragma omp master

        if(tid==0)
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
            printf("Heloo max = %d\n", maxth);
        }
    }

#pragma omp parallel private(tid)
    {
        /* Obtain thread number */
        //omp_set_num_threads(4);
        tid = omp_get_thread_num();
        maxth=omp_get_max_threads();
        printf("Hello World from another thread = %d\n", tid);

        /* Only master thread does this */

        // #pragma omp master

        if(tid==0)
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
            printf("Heloo max = %d\n", maxth);
        }
    }
    exit(0);
}
