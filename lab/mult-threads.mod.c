#ifdef _POMP
#  undef _POMP
#endif
#define _POMP 200110

#include "mult-threads.c.opari.inc"
#line 1 "mult-threads.c"
#include <stdlib.h>

#define N 1000

int *a;
int *b;
int *c;

int main()
{
  int i, j, k;
  int acc;

  a = (int *)malloc(N*N*sizeof(int));
  b = (int *)malloc(N*N*sizeof(int));
  c = (int *)malloc(N*N*sizeof(int));

  for(i = 0; i < N; i++)
    for(j = 0; j < N; j++){
      a[i * N + j] = rand();
      b[i * N + j] = rand();
    }
      

  for(i = 0; i < N; i++)
    for(j = 0; j < N; j++){
      acc = 0;
POMP_Parallel_fork(&omp_rd_1);
#line 29 "mult-threads.c"
#pragma omp parallel     reduction(+:acc)
{ POMP_Parallel_begin(&omp_rd_1);
POMP_For_enter(&omp_rd_1);
#line 29 "mult-threads.c"
#pragma omp          for                  nowait
      for(k = 0; k < N; k++)
	acc += a[i * N + k] * b[k * N + j];
POMP_Barrier_enter(&omp_rd_1);
#pragma omp barrier
POMP_Barrier_exit(&omp_rd_1);
POMP_For_exit(&omp_rd_1);
POMP_Parallel_end(&omp_rd_1); }
POMP_Parallel_join(&omp_rd_1);
#line 32 "mult-threads.c"
      c[i * N + j] = acc;
    }
}
      
