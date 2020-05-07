/**************************Declarations**************************/
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>
#include <mpi.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef struct entryA
{
  int user;
  int item;
  double rate;
  double recom;
  struct entryA *nextItem;
  struct entryA *nextUser;
} entryA;

entryA *createNode();
void alloc_A(int nU, int nI, entryA***_A_user, entryA***_A_item,
             entryA***_A_user_aux, entryA***_A_item_aux);



/****************************************************************/

int main(int argc, char *argv[])
{
  FILE *fp;
  int nIter, nFeat, nUser, nItem, nEntry;
  int *solution;
  double deriv = 0;
  double alpha, sol_aux;
  double elapsed_time;
  double **L, **R, **B, **newL, **newR;
  char *outputFile;

  entryA **A_user, **A_user_aux, **A_item, **A_item_aux;
  entryA *A_aux1, *A_aux2;

  MPI_Status status;
  int id, np,
      i, rounds;
  double secs;

  MPI_Init(&argc, &argv);
  MPI_Barrier(MPI_COMM_WORLD);
  elapsed_time = -MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (argc != 2)
  {
    printf("error: command of type ./matFact <filename.in>\n");
    MPI_Finalize();
    exit(1);
  }

  fp = fopen(argv[1], "r");
  if (fp == NULL)
  {
    printf("error: cannot open file\n");
    MPI_Finalize();
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
    // load of entryAof matrix A
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

  /************Code for load balance************/
  typedef struct group{
    // total count of number of entries
    int count;
    // indexes of users assigned to group
    int idx[nUser];
  } group;

  // number of groups
  double divf = sqrt(np);
  int div = floor(div);
  // number of entries per division rounded down
  int lower = nEntry / div;
  lower = (int)floor(lower);
  // number of entries per division plus one
  int upper = nEntry / div + 1;
  // rest of division
  int rest =  div % nEntry;

  int aux;
  
  // vector with groups of users 
  group *groups = (group*)calloc(sizeof(group), div);
  // vector with number of items per user
  int *counts = (int*)calloc(sizeof(int), nUser);

  int j = 0, k = 0;

  for(i = 0; i < nUser; i++){
    // if we are in the last partition it gets the remaing users
    if(j = div-1){
      groups[j].count + counts[i];
      groups[j].idx[k] = i;
      k++;
      continue;
    }
    else{
      // aux that stores the possibel new user for that group
      aux = groups[j].count + counts[i];

      if(aux <= lower){
        groups[j].count = aux;
        groups[j].idx[k] = i;
        k++;
        continue;
      }
      // if threshold is atchived
      else if((aux - upper) >= 0){
        // check to see where the difference betwhen the two threshold
        // is less
        if((aux - lower) < (aux - upper)){
          // add the difference between the desired threshold
          rest += aux - lower;
          // advance to next group
          j++;
          k = 0;
        }
        // check to see if there were already other groups to be above the upper threshold
        // decrising already the rest
        else if(rest > (aux - upper)){
          // subtract the difference between the desired threshold
          rest -= aux - upper;
          // add the final user for that group
          groups[j].count = aux;
          groups[j].idx[k] = i;
          // advance to next group
          j++;
          k = 0;
        }
        // if so does not add the next user to the group
        else{
          // add the difference between the desired threshold
          rest += aux - lower;
          // advance to next group
          j++;
          k = 0;
        }
      }
    }
  }

  /****************************End Setup****************************/

  /***********************Matrix Factorization**********************/

  /************************Write in Terminal************************/

  /**************************Free of memory*************************/

  free(groups);
  free(counts)

  elapsed_time += MPI_Wtime();
  MPI_Finalize();
  return 0;
}

entryA*createNode() {

  entryA*A;
  A = (entryA*)malloc(sizeof(entryA));
  A->nextItem = NULL;
  A->nextUser = NULL;

  return A;
}

void alloc_A(int nU, int nI, entryA***_A_user, entryA***_A_item,
             entryA***_A_user_aux, entryA***_A_item_aux) {

  *_A_user = (entryA**)calloc(sizeof(entryA*), nU);
  *_A_item = (entryA**)calloc(sizeof(entryA*), nI);

  *_A_user_aux = (entryA**)calloc(sizeof(entryA*), nU);
  *_A_item_aux = (entryA**)calloc(sizeof(entryA*), nI);
}
