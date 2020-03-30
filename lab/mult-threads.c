#include <omp.h> 
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
#pragma omp parallel for reduction(+:acc)
      for(k = 0; k < N; k++)
	acc += a[i * N + k] * b[k * N + j];
      c[i * N + j] = acc;
    }
}
      
