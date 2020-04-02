/**************************Declarations**************************/

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef struct entryA {
  int user;
  int item;
  double rate;
  double recom;
  struct entryA *nextItem;
  struct entryA *nextUser;
} entryA;

void alloc_A(int nU, int nI, entryA ***_A_user, entryA ***_A_item,
             entryA ***_A_user_aux, entryA ***_A_item_aux);

entryA *createNode();
void random_fill_LR(int nU, int nI, int nF, double ***L, double ***R,
                    double ***newL, double ***newR);
void alloc_LRB(int nU, int nI, int nF, double ***L, double ***R, double ***newL,
               double ***newR, double ***B);
void multiply_LR(int nU, int nF, double ***L, double ***R, double ***newL, 
                  double ***newR, entryA ***A_user);
void update_LR(double ***L, double ***R, double ***newL, double ***newR);
void free_LR(int nU, int nF, double ***L, double ***R, double ***newL,
             double ***newR, double ***B);

/****************************************************************/

int main(int argc, char *argv[]) {
  FILE *fp;
  int nIter, nFeat, nUser, nItem, nEntry, B_item;
  int *solution;
  double deriv = 0;
  double alpha, sol_aux;
  double **L, **R, **B, **newL, **newR;
  char *outputFile;

  entryA **A_user, **A_user_aux, **A_item, **A_item_aux;
  entryA *A_aux1, *A_aux2;

  if (argc != 2) {
    printf("error: command of type ./matFact <filename.in>\n");
    exit(1);
  }

  fp = fopen(argv[1], "r");
  if (fp == NULL) {
    printf("error: cannot open file\n");
    exit(1);
  }

  /******************************Setup******************************/
  // Functional Parallelism
  fscanf(fp, "%d", &nIter);
  fscanf(fp, "%lf", &alpha);
  fscanf(fp, "%d", &nFeat);
  fscanf(fp, "%d %d %d", &nUser, &nItem, &nEntry);

  alloc_A(nUser, nItem, &A_user, &A_item, &A_user_aux, &A_item_aux);

  solution = (int *)malloc(sizeof(int) * nUser);

  // Fill A
  for (int i = 0; i < nEntry; i++) {
    A_aux1 = createNode();
    fscanf(fp, "%d %d %lf", &(A_aux1->user), &(A_aux1->item), &(A_aux1->rate));

    if (A_user[A_aux1->user] == NULL) {

      A_user[A_aux1->user] = A_aux1;
      A_user_aux[A_aux1->user] = A_aux1;

    } else {

      A_user_aux[A_aux1->user]->nextItem = A_aux1;
      A_user_aux[A_aux1->user] = A_aux1;
    }

    if (A_item[A_aux1->item] == NULL) {

      A_item[A_aux1->item] = A_aux1;
      A_item_aux[A_aux1->item] = A_aux1;

    } else {

      A_item_aux[A_aux1->item]->nextUser = A_aux1;
      A_item_aux[A_aux1->item] = A_aux1;
    }
  }

  fclose(fp);
  free(A_item_aux);
  free(A_user_aux);

  alloc_LRB(nUser, nItem, nFeat, &L, &R, &newL, &newR, &B);
  random_fill_LR(nUser, nItem, nFeat, &L, &R, &newL, &newR);
  multiply_LR(nUser, nFeat, &L, &R, &newL, &newR,&A_user);

  /****************************End Setup****************************/

  /***********************Matrix Factorization**********************/
  
  for (int n = 0; n < nIter; n++) {
    // Matrix L
    #pragma omp parallel default(none) shared(nUser, nItem, nFeat, A_user, A_item, R, L,newR, newL, alpha, deriv, A_aux1)
    {
      #pragma omp for firstprivate(A_aux1, deriv) nowait schedule(dynamic)
      for (int i = 0; i < nUser; i++) {
        for (int k = 0; k < nFeat; k++) {

          A_aux1 = A_user[i];
          while (A_aux1 != NULL) {
            deriv +=
                2 * (A_aux1->rate - A_aux1->recom) * (-R[k][A_aux1->item]);
            A_aux1 = A_aux1->nextItem;
          }

          newL[i][k] = L[i][k] - alpha * deriv;
          deriv = 0;
        }
      }

      // Matrix R
      #pragma omp for firstprivate(A_aux1, deriv) nowait schedule(dynamic)
      for (int j = 0; j < nItem; j++) {
        for (int k = 0; k < nFeat; k++) {

          A_aux1 = A_item[j];
          while (A_aux1 != NULL) {
            deriv +=
                2 * (A_aux1->rate - A_aux1->recom) * (-L[A_aux1->user][k]);
            A_aux1 = A_aux1->nextUser;
          }

          newR[k][j] = R[k][j] - alpha * deriv;
          deriv = 0;
        }
      }
    }
    update_LR(&L, &R, &newL, &newR);
    multiply_LR(nUser, nFeat, &L, &R, &newL, &newR,&A_user);
  }
  /*********************End Matrix Factorization********************/
 
  #pragma omp parallel for //collapse(2)
  for (int i = 0; i < nUser; i++){
    for (int j = 0; j < nItem; j++){
      B[i][j] = 0;
      
      for (int k = 0; k < nFeat; k++)
        B[i][j] += L[i][k] * R[k][j];
    }
  }

  #pragma omp parallel default(none) private(B_item, sol_aux, A_aux1) shared(B, solution, nUser, A_user, nItem)
  {
    #pragma omp for schedule(dynamic)
    for (int k = 0; k < nUser; k++) {
      B_item = 0;
      sol_aux = 0;
      A_aux1 = A_user[k];

      while (A_aux1 != NULL) {
        B[k][A_aux1->item] = 0;
        A_aux1 = A_aux1->nextItem;
      }

      while (B_item < nItem) {
        if (B[k][B_item] > sol_aux) {
          solution[k] = B_item;
          sol_aux = B[k][B_item];
        }

        B_item++;
      }
    }
  }
  

  /****************************Write File***************************/
  outputFile = strtok(argv[1], ".");
  strcat(outputFile, ".out\0");

  fp = fopen(outputFile, "w");
  if (fp == NULL) {
    printf("error: cannot open file\n");
    exit(1);
  }

  for (int i = 0; i < nUser; i++) {
    fprintf(fp, "%d\n", solution[i]);
  }

  fclose(fp);
  /*****************************************************************/

  /******************************Free A*****************************/
  for (int i = 0; i < nUser; i++) {

    A_aux1 = A_user[i];

    while (A_aux1 != NULL) {
      A_aux2 = A_aux1->nextItem;
      free(A_aux1);
      A_aux1 = A_aux2;
    }
  }
  free(A_user);
  free(A_item);
  /*****************************************************************/
  free(solution);
  free_LR(nUser, nFeat, &L, &R, &newL, &newR, &B);

  return 0;
}

void alloc_A(int nU, int nI, entryA ***_A_user, entryA ***_A_item,
             entryA ***_A_user_aux, entryA ***_A_item_aux) {

  *_A_user = (entryA **)calloc(sizeof(entryA *), nU);
  *_A_item = (entryA **)calloc(sizeof(entryA *), nI);

  *_A_user_aux = (entryA **)calloc(sizeof(entryA *), nU);
  *_A_item_aux = (entryA **)calloc(sizeof(entryA *), nI);
}

entryA *createNode() {

  entryA *A;
  A = (entryA *)malloc(sizeof(entryA));
  A->nextItem = NULL;
  A->nextUser = NULL;

  return A;
}

void alloc_LRB(int nU, int nI, int nF, double ***L, double ***R, double ***newL,
               double ***newR, double ***B) {
                 
  *B = (double **)malloc(sizeof(double *) * nU);
  *L = (double **)malloc(sizeof(double *) * nU);
  *newL = (double **)malloc(sizeof(double *) * nU);
  *R = (double **)malloc(sizeof(double *) * nF);
  *newR = (double **)malloc(sizeof(double *) * nF);
  
  #pragma omp parallel for
  for (int i = 0; i < nU; i++) {
    (*B)[i] = (double *)malloc(sizeof(double) * nI);
    (*L)[i] = (double *)malloc(sizeof(double) * nF);
    (*newL)[i] = (double *)malloc(sizeof(double) * nF);
  }

  #pragma omp parallel for
  for (int i = 0; i < nF; i++) {
    (*R)[i] = (double *)malloc(sizeof(double) * nI);
    (*newR)[i] = (double *)malloc(sizeof(double) * nI);
  }
}

void multiply_LR(int nU, int nF, double ***L, double ***R, double ***newL, 
                  double ***newR, entryA ***A_user) {
  entryA *A_aux1;

  #pragma omp parallel
  {
    #pragma omp for private(A_aux1) schedule(dynamic)
    for (int i = 0; i < nU; i++) {
      A_aux1 = (*A_user)[i];
      while (A_aux1 != NULL) {
        A_aux1->recom = 0;

        for (int k = 0; k < nF; k++)
          A_aux1->recom += (*L)[i][k] * (*R)[k][A_aux1->item];
        A_aux1 = A_aux1->nextItem;
      }
    }
  }
}

void random_fill_LR(int nU, int nI, int nF, double ***L, double ***R,
                    double ***newL, double ***newR) {
  srandom(0);
  // Data Parallelism
  for (int i = 0; i < nU; i++)
    for (int j = 0; j < nF; j++) {
      (*L)[i][j] = RAND01 / (double)nF;
      (*newL)[i][j] = (*L)[i][j];
    }
  // Data Parallelism
  for (int i = 0; i < nF; i++)
    for (int j = 0; j < nI; j++) {
      (*R)[i][j] = RAND01 / (double)nF;
      (*newR)[i][j] = (*R)[i][j];
    }
}

void update_LR(double ***L, double ***R, double ***newL, double ***newR) {

  double **aux;
  aux = *L;
  *L = *newL;
  *newL = aux;

  aux = *R;
  *R = *newR;
  *newR = aux;
}

void free_LR(int nU, int nF, double ***L, double ***R, double ***newL,
             double ***newR, double ***B) {

  // Data Parallelism
  for (int i = 0; i < nU; i++) {
    free((*B)[i]);
    free((*L)[i]);
    free((*newL)[i]);
  }
  free(*B);
  free(*L);
  free(*newL);

  // Data Parallelism
  for (int i = 0; i < nF; i++) {
    free((*R)[i]);
    free((*newR)[i]);
  }
  free(*newR);
  free(*R);
}
