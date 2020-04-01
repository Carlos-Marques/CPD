#ifdef _POMP
#  undef _POMP
#endif
#define _POMP 200110

#include "mult-good.c.opari.inc"
#line 1 "mult-good.c"
#include <stdlib.h>

#define N 1000

int *a;
int *b;
int *c;

int main()
{
  int i, j, k;


  a = (int *)malloc(N*N*sizeof(int));
  b = (int *)malloc(N*N*sizeof(int));
  c = (int *)malloc(N*N*sizeof(int));

  for(i = 0; i < N; i++)
    for(j = 0; j < N; j++){
      a[i * N + j] = rand();
      b[i * N + j] = rand();
    }
      

POMP_Parallel_fork(&omp_rd_1);
#line 26 "mult-good.c"
#pragma omp parallel     private(i,j,k)
{ POMP_Parallel_begin(&omp_rd_1);
POMP_For_enter(&omp_rd_1);
#line 26 "mult-good.c"
#pragma omp          for                nowait
  for(i = 0; i < N; i++)
    for(j = 0; j < N; j++)
      for(k = 0; k < N; k++)
	c[i * N + j] += a[i * N + k] * b[k * N + j];
POMP_Barrier_enter(&omp_rd_1);
#pragma omp barrier
POMP_Barrier_exit(&omp_rd_1);
POMP_For_exit(&omp_rd_1);
POMP_Parallel_end(&omp_rd_1); }
POMP_Parallel_join(&omp_rd_1);
#line 31 "mult-good.c"


}
