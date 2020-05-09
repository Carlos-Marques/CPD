/**************************Declarations**************************/

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef struct entry {
  int user;
  int item;
  double rate;
  double recom;
  struct entry *nextItem;
  struct entry *nextUser;
} entry;


void alloc_A(int nU, int nI, entry ***_A_user, entry ***_A_item,
             entry ***_A_user_aux, entry ***_A_item_aux);

entry *createNode();

void alloc_LRB(int nU, int nI, int nF, double ***L, double ***R, double ***newL,
               double ***newR, double ***B);
void random_fill_LR(int nU, int nI, int nF, double ***L, double ***R,
                    double ***newL, double ***newR);
void update_recom(int nU, int nF, double ***L, double ***R,
                 entry ***A_user);

void update_LR(double ***L, double ***R, double ***newL, double ***newR);
void free_LR(int nU, int nF, double ***L, double ***R, double ***newL,
             double ***newR, double ***B);

/****************************************************************/

int main(int argc, char *argv[]) {
  FILE *fp;
  int nIter, nFeat, nUser, nItem, nEntry;
  int *solution;
  double deriv = 0;
  double alpha, sol_aux;
  double **L, **R, **B, **newL, **newR;
  char *outputFile;

  entry **A_user, **A_user_aux, **A_item, **A_item_aux;
  entry *A_aux1, *A_aux2;

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
  // read of first parameters of file
  fscanf(fp, "%d", &nIter);
  fscanf(fp, "%lf", &alpha);
  fscanf(fp, "%d", &nFeat);
  fscanf(fp, "%d %d %d", &nUser, &nItem, &nEntry);

  // alloc struct that holds A and it's approximation, B
  alloc_A(nUser, nItem, &A_user, &A_item, &A_user_aux, &A_item_aux);

  // alloc vector that holds highest recom. per user
  solution = (int *)malloc(sizeof(int) * nUser);

  // construct of a list of lists
  for (int i = 0; i < nEntry; i++) {
    A_aux1 = createNode();
    // load of entry of matrix A
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

  // alloc L, R and B where B is only used for the final calculation
  alloc_LRB(nUser, nItem, nFeat, &L, &R, &newL, &newR, &B);
  // init L and R with random values
  random_fill_LR(nUser, nItem, nFeat, &L, &R, &newL, &newR);
  // init of values of B that are to be approximated to the rate of 
  // items per user
  update_recom(nUser, nFeat, &L, &R, &A_user);


  /****************************End Setup****************************/

  /***********************Matrix Factorization**********************/
  
  // main loop with stopping criterium
  for (int n = 0; n < nIter; n++) {
    // calculation of the t+1 iteration of L 
    for (int i = 0; i < nUser; i++) {
      for (int k = 0; k < nFeat; k++) {

        A_aux1 = A_user[i];
        // sum of derivatives per item
        while (A_aux1 != NULL) {
          deriv +=
              2 * (A_aux1->rate - A_aux1->recom) * (-R[k][A_aux1->item]);
          A_aux1 = A_aux1->nextItem;
        }
        // final calculation of t+1
        newL[i][k] = L[i][k] - alpha * deriv;
        deriv = 0;
      }
    }

    // calculation of the t+1 iteration of R
    for (int j = 0; j < nItem; j++) {
      for (int k = 0; k < nFeat; k++) {

        A_aux1 = A_item[j];
        // sum of derivatives per user
        while (A_aux1 != NULL) {
          deriv +=
              2 * (A_aux1->rate - A_aux1->recom) * (-L[A_aux1->user][k]);
          A_aux1 = A_aux1->nextUser;
        }
        // final calculation of t+1
        newR[k][j] = R[k][j] - alpha * deriv;
        deriv = 0;
      }
    }
    // update of L and R with the t+1 values
    update_LR(&L, &R, &newL, &newR);
    // update of B for each non-zero element of A
    update_recom(nUser, nFeat, &L, &R, &A_user);

  }
  /*********************End Matrix Factorization********************/
  
  // calculation of the entire B matrix meaning the 
  // internal product between L and R
  for (int i = 0; i < nUser; i++)
    for (int j = 0; j < nItem; j++) {
      B[i][j] = 0;
      for (int k = 0; k < nFeat; k++)
        B[i][j] += L[i][k] * R[k][j];
    }

  // update of solution with highest calculated recomendation 
  // rate per user 
  for (int k = 0; k < nUser; k++) {
    sol_aux = 0;
    A_aux1 = A_user[k];

    // update entry of B to 0 if item already rated
    while (A_aux1 != NULL) {
      B[k][A_aux1->item] = 0;
      A_aux1 = A_aux1->nextItem;
    }
    // save item with highest rate
    for(int j = 0; j < nItem; j++){
      if (B[k][j] > sol_aux) {
        solution[k] = j;
        sol_aux = B[k][j];
      }
    }
  }

  /****************************Write File***************************/
  
  // create .out file for writing
  outputFile = strtok(argv[1], ".");
  strcat(outputFile, ".out\0");

  fp = fopen(outputFile, "w");
  if (fp == NULL) {
    printf("error: cannot open file\n");
    exit(1);
  }

  // write recomendation on file per user
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

void alloc_A(int nU, int nI, entry ***_A_user, entry ***_A_item,
             entry ***_A_user_aux, entry ***_A_item_aux) {

  *_A_user = (entry **)calloc(sizeof(entry *), nU);
  *_A_item = (entry **)calloc(sizeof(entry *), nI);

  *_A_user_aux = (entry **)calloc(sizeof(entry *), nU);
  *_A_item_aux = (entry **)calloc(sizeof(entry *), nI);
}

entry *createNode() {

  entry *A;
  A = (entry *)malloc(sizeof(entry));
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

  for (int i = 0; i < nU; i++) {
    (*B)[i] = (double *)malloc(sizeof(double) * nI);
    (*L)[i] = (double *)malloc(sizeof(double) * nF);
    (*newL)[i] = (double *)malloc(sizeof(double) * nF);
  }

  for (int i = 0; i < nF; i++) {
    (*R)[i] = (double *)malloc(sizeof(double) * nI);
    (*newR)[i] = (double *)malloc(sizeof(double) * nI);
  }
}

void random_fill_LR(int nU, int nI, int nF, double ***L, double ***R,
                    double ***newL, double ***newR) {
  srandom(0);
  
  // init of L, stable version, and newL for t+1
  for (int i = 0; i < nU; i++)
    for (int j = 0; j < nF; j++) {
      (*L)[i][j] = RAND01 / (double)nF;
      (*newL)[i][j] = (*L)[i][j];
    }

  // init of R, stable version, and newR for t+1
  for (int i = 0; i < nF; i++)
    for (int j = 0; j < nI; j++) {
      (*R)[i][j] = RAND01 / (double)nF;
      (*newR)[i][j] = (*R)[i][j];
    }
}


void multiply_LR(int nU, int nF, double ***L, double ***R,
                 entryA ***A_user) {
  entryA *A_aux1;

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


void update_recom(int nU, int nF, double ***L, double ***R,
                 entry ***A_user) {
  entry *A_aux1;

  // update recomendation for all non-zero entries meaning
  // the approximation of B to A
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

void free_LR(int nU, int nF, double ***L, double ***R, double ***newL,
             double ***newR, double ***B) {

  for (int i = 0; i < nU; i++) {
    free((*B)[i]);
    free((*L)[i]);
    free((*newL)[i]);
  }
  free(*B);
  free(*L);
  free(*newL);

  for (int i = 0; i < nF; i++) {
    free((*R)[i]);
    free((*newR)[i]);
  }
  free(*newR);
  free(*R);
}
