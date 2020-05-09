/**************************Declarations**************************/
#include <mpi.h>
#include <math.h>
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
  int recom;
  struct entry *nextItem;
  struct entry *nextUser;
} entry;

 typedef struct group{
    // total count of number of entries
    int count;
    // first and last user of that group
    int firstUser;
    int lastUser;
    // list of machines for that group
    int machines[200];
    // total machines
    int numIdx;
  } group;

void alloc_A(int nU, int nI, entry ***_A_user, entry ***_A_item,
             entry ***_A_user_aux, entry ***_A_item_aux);

void update_LR(double ***L, double ***R, double ***newL, double ***newR);

entry *createNode();

void update_recom(int nU, int k0, int k1, double ***L, double ***R,
                 entry ***A_user, double *B, double *group_B, MPI_Comm group_comm, int sizeof_B);

void alloc_LRB(int nU, int nI, int nF, double ***L, double ***R, double ***newL,
               double ***newR, double ***B);

void random_fill_LR(int nU, int nI, int nF, double ***L, double ***R,
                    double ***newL, double ***newR);

entry** split_A(entry ***_A_user, entry ***_A_item, int new_first_user, int interval, int nU, int nI);

void free_LR(int nU, int nF, double ***L, double ***R, double ***newL,
             double ***newR, double ***B);

/****************************************************************/

int main(int argc, char *argv[]) {
  FILE *fp;
  int nIter, nFeat, nUser, nItem, nEntry, nUser_original;
  int *solution;
  double deriv = 0;
  double alpha, sol_aux;
  double **L, **R, **B, **newL, **newR, *local_B, *group_B, *local_B_final, *group_B_final;
  char *outputFile;
  double elapsed_time;

  entry **A_user, **A_user_aux, **A_item, **A_item_aux;
  entry **A_user_partition;
  entry *A_aux1, *A_aux2;

  MPI_Status status;
  MPI_Init (&argc, &argv);
  MPI_Barrier(MPI_COMM_WORLD);
  elapsed_time = -MPI_Wtime();
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  if (argc != 2) {
    printf("error: command of type ./matFact <filename.in>\n");
    MPI_Finalize();
    exit(1);
  }

  fp = fopen(argv[1], "r");
  if (fp == NULL) {
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
  nUser_original=nUser;

  // alloc struct that holds A and it's approximation, B
  alloc_A(nUser, nItem, &A_user, &A_item, &A_user_aux, &A_item_aux);

  // alloc vector that holds highest recom. per user
  solution = calloc(sizeof(int), nUser);

  // vector with number of items per user
  int *count = (int*)calloc(sizeof(int), nUser);
  int auxUser = 0, userIdx = 0;

  // construct of a list of lists
  for (int i = 0; i < nEntry; i++) {
    A_aux1 = createNode();
    // load of entryAof matrix A
    fscanf(fp, "%d %d %lf", &(A_aux1->user), &(A_aux1->item), &(A_aux1->rate));

    // store the sum of items per user
    if(auxUser == A_aux1->user){
      count[userIdx] ++;
    }
    else{
      auxUser = A_aux1->user;
      userIdx++;
      count[userIdx] ++;
    }

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

  /****************************End Setup****************************/
  //A_user = split_A(&A_user, &A_item, 1, 2, nUser, nItem);
  //nUser=2;   //actualizar o n_user para o numero de users com o qual este px efecticamente trabalha

  alloc_LRB(nUser, nItem, nFeat, &L, &R, &newL, &newR, &B);
  random_fill_LR(nUser, nItem, nFeat, &L, &R, &newL, &newR);

  /*******************Code for load balance********************/
  // number of groups
  int div = floor(sqrt(world_size));
  // number of entries per division rounded down
  int lower = nEntry / div;
  // number of entries per division plus one
  int upper = nEntry / div + 1;
  // rest of division
  int rest =  div % nEntry;
  
  // lower num of machines per group
  int lowerMach = world_size / div;
  // upper num of machines per group
  int upperMach = lowerMach + 1;
  // only use upper if rest of division not zero
  int restMach = world_size % div;

  int aux = 0, j = 0;

  // vector with groups of users 
  group *groups = (group*)calloc(sizeof(group), div);

  for(int i = 0; i < nUser; i++){
    // save first user of the group
    if(groups[j].count == 0) groups[j].firstUser = i;
    
    // if we are in the last partition it gets the remaing users
    if(j == div-1) groups[j].count += count[i];
    else{
      // aux that stores the possibel new user for that group
      aux = groups[j].count + count[i];

      // if the lower threshold is not with the sum of the new user add it and continue
      if(aux <= lower) groups[j].count = aux;
      // if threshold is atchived
      else if((aux - upper) >= 0){
        // check to see where the difference betwhen the two threshold
        // is less
        if(abs(groups[j].count - lower) < (aux - upper)){
          // add the difference between the desired threshold
          rest += aux - lower;
          // save last user of the group
          groups[j].lastUser = i-1;
          // advance to next group
          j++;
          // the count that wasnt considered becaused it was closer to the lower boundary
          // is put automatically in the next group 
          groups[j].count = count[i];
          // save first user of the group
          groups[j].firstUser = i;
        }
        // check to see if there were already other groups to be above the upper threshold
        // decrising already the rest
        else if(rest > (aux - upper)){
          // subtract the difference between the desired threshold
          rest -= aux - upper;
          // add the final user for that group
          groups[j].count = aux;
          // save last user of the group
          groups[j].lastUser = i;
          // advance to next group
          j++;
        }
        // if so, it does not add the next user to the group
        else{
          // add the difference between the desired threshold
          rest += aux - lower;
          // add the final user for that group
          groups[j].count = aux;
          // save last user of the group
          groups[j].lastUser = i;
          // advance to next group
          j++;
        }
      }
    }
    // if we are in the last iteration
    if(i+1 == nUser) groups[j].lastUser = i;
  }

  int k = 0;
  int divMach = upperMach;

  // assigns a each group a set of computers 
  for(int i = 0; i < div; i++){
    if(restMach == 0) divMach = lowerMach;
    else restMach -= 1;

    for(int j = 0; j < divMach; j++){
      // if de problem in not divisible by the number of computers sets
      // to the maximun possible meaning the group with zero rows are not 
      // assinged to any machine
      if(groups[i].count != 0){
        groups[i].machines[j] = k;
        k++;
      }
      else break;
    }
  }
  
  //printf("\ngroup %d - count: %d - first user: %d - last user: %d\n", i, groups[i].count, groups[i].firstUser, groups[i].lastUser);
  //printf("lower:%d - upper:%d - rest:%d\n", lowerMach, upperMach, restMach);
  //if(!world_rank) printf("user:%d - lower:%d - upper:%d - rest:%d\n", nEntry, lower, upper, rest);
  //fflush(stdin);
  //printf("  m: %d\n", k);
  //fflush(stdin);

  /*******************Code for load balance********************/



  MPI_Comm B_comm, R_comm;
  MPI_Group world_group, B_group, R_group;

  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  int k0, k1;
  int counter = 0;
  int user_i = 0;
  int user_l = 0;

  //printf("world_rank: %d\n", world_rank);
  if(world_rank == 0 || world_rank == 1){
    int rank[] = {0, 1};
    MPI_Group_incl(world_group, 2, rank, &B_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, B_group, 0, &B_comm);
    nUser = 500;
    user_i = 0;
    A_user = split_A(&A_user, &A_item, user_i, nUser, nUser_original, nItem);


    for (int i = 0; i < nUser; i++) {
      A_aux1 = A_user[i];

      while(A_aux1 != NULL){
        A_aux1->recom = counter;
        counter++;

        A_aux1 = A_aux1->nextItem;
      }
    }

    local_B = malloc(sizeof(double) * counter+1);
    group_B = malloc(sizeof(double)* counter+1);

/*
    if(world_rank == 0) {
      update_recom(nUser, 0, 1, &L, &R, &A_user, local_B, group_B, B_comm);
    }
    else{
      update_recom(nUser, 1, 2, &L, &R, &A_user, local_B, group_B, B_comm);
    }
*/
  }
  else {
    int rank[] = {2, 3};
    MPI_Group_incl(world_group, 2, rank, &B_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, B_group, 0, &B_comm);
    nUser = 500;
    user_i = 500;
    A_user = split_A(&A_user, &A_item, user_i, nUser, nUser_original, nItem);

    for (int i = 0; i < nUser; i++) {
      A_aux1 = A_user[i];

      while(A_aux1 != NULL){
        A_aux1->recom = counter;
        counter++;

        A_aux1 = A_aux1->nextItem;
      }
    }


    local_B = malloc(sizeof(double) * counter+1);
    group_B = malloc(sizeof(double)* counter+1);
/*
    if(world_rank == 2) {
      update_recom(nUser, 0, 1, &L, &R, &A_user, local_B, group_B, B_comm);
    }
    else {
      update_recom(nUser, 1, 2, &L, &R, &A_user, local_B, group_B, B_comm);
    }
*/

  }

/*
  if(world_rank == 0){
    printf("\n\n Visto pelos Users:\n");
    for (int i = 0; i < nUser; i++)
    {
    A_aux1 = A_user[i];
    printf("\n\n User %d ligado a:\n", i);
    while (A_aux1 != NULL) {
      printf("user: %d item: %d rate: %f recom: %i\n",A_aux1->user, A_aux1->item, A_aux1->rate, A_aux1->recom);
      A_aux1 = A_aux1->nextItem;
    }
    printf("\n");
  }

  printf("\n\n Visto pelos Items:");
  for (int i = 0; i < nItem; i++)
  {
    A_aux1 = A_item[i];
    printf("\n\n Item %d ligado a:\n", i);
    while (A_aux1 != NULL) {
      printf("user: %d item: %d rate: %f recom: %i\n",A_aux1->user, A_aux1->item, A_aux1->rate, A_aux1->recom);
      A_aux1 = A_aux1->nextUser;
    }
    printf("\n");
  }
    }
*/

  if(world_rank == 0 || world_rank == 2) {
    int rank[] = {0, 2};
    MPI_Group_incl(world_group, 2, rank, &R_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, R_group, 0, &R_comm);
    k0 = 0;
    k1 = 50;
  }
  else {
    int rank[] = {1, 3};
    MPI_Group_incl(world_group, 2, rank, &R_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, R_group, 0, &R_comm);
    k0 = 50;
    k1 = 100;
  }

  int inter = k1-k0;
  int sizeof_derivs = (inter*nItem);
  double *derivs = malloc(sizeof(double) * sizeof_derivs);
  double *global_derivs = malloc(sizeof(double) * sizeof_derivs);

  //printf("HERE %d\n", world_rank);
  for (int n = 0; n < nIter; n++) {

  update_recom(nUser, k0, k1, &L, &R, &A_user, local_B, group_B, B_comm, counter+1);

  //printf("Calc B %d\n", world_rank);

  /*
  for(int i = 0; i < counter; i++){
    printf("id: %d B:%f\n", i, group_B[i]);
  }
  */
  int test = 0;


  //L+1
  for (int i = 0; i < nUser; i++) {
      for (int k = k0; k < k1; k++) {

        A_aux1 = A_user[i];
        // sum of derivatives per item

        while (A_aux1 != NULL) {
          deriv +=
              2 * (A_aux1->rate - group_B[A_aux1->recom]) * (-R[k][A_aux1->item]);
          A_aux1 = A_aux1->nextItem;
        }
        // final calculation of t+1
        A_aux1 = A_user[i];
        if(A_aux1 != NULL)
          newL[A_aux1->user][k] = L[A_aux1->user][k] - alpha * deriv;
        //printf("L: %f\n", L[A_aux1->user][k]);
        //printf("newL: %f\n", newL[A_aux1->user][k]);
        deriv = 0;
      }
      //printf("\n");
    }

  //printf("Calc L+1 %d\n", world_rank);
  //R+1
  int c = 0;
  for (int j = 0; j < nItem; j++) {
      for (int k = k0; k < k1; k++) {

        A_aux1 = A_item[j];

        // sum of derivatives per user
        while (A_aux1 != NULL) {
          deriv += 2 * (A_aux1->rate - group_B[A_aux1->recom]) * (-L[A_aux1->user][k]);
          A_aux1 = A_aux1->nextUser;
        }
        // final calculation of t+1
        derivs[c] = alpha * deriv;
        c++;
        deriv = 0;
      }
    }

  /*printf("sizeof_derivs: %d\n", c);
  printf("cal size: %d\n", sizeof_derivs);
  */
  MPI_Allreduce(derivs, global_derivs, sizeof_derivs, MPI_DOUBLE, MPI_SUM, R_comm);

  c = 0;
  for (int j = 0; j < nItem; j++) {
    for (int k = k0; k < k1; k++) {

      newR[k][j] = R[k][j] - global_derivs[c];
      //printf("world_rank: %d global_derivs: %f R: %f newR: %f\n", world_rank, global_derivs[c], R[k][j], newR[k][j]);
      c++;
      }
  }

  update_LR(&L, &R, &newL, &newR);
  }


  local_B_final = malloc(sizeof(double) * nUser * nItem);
  group_B_final = malloc(sizeof(double) * nUser * nItem);

  int c = 0;
  for (int i = 0; i < nUser; i++) {
    user_l = user_i + i;
    for (int j = 0; j < nItem; j++) {
      local_B_final[c] = 0;
      group_B_final[c] = 0;
      for (int k = k0; k < k1; k++){
        local_B_final[c] += L[user_l][k] * R[k][j];
      }
      c++;
    }
  }

  MPI_Reduce(local_B_final, group_B_final, nUser * nItem, MPI_DOUBLE, MPI_SUM, 0, B_comm);

  c = 0;
  for(int i = 0; i < nUser; i++) {
    user_l = user_i + i;
    for(int j = 0; j < nItem; j++) {
      B[user_l][j] = group_B_final[c];
      c++;
    }
  }

  for (int k = 0; k < nUser; k++) {
    sol_aux = 0;
    A_aux1 = A_user[k];

    // update entry of B to 0 if item already rated
    while (A_aux1 != NULL) {
      B[A_aux1->user][A_aux1->item] = 0;
      A_aux1 = A_aux1->nextItem;
    }

    user_l = user_i + k;
    // save item with highest rate
    for(int j = 0; j < nItem; j++){
      if (B[user_l][j] > sol_aux) {
        solution[user_l] = j;
        sol_aux = B[user_l][j];
      }
    }
  }

  int *global_solution;
  global_solution = calloc(sizeof(int), nUser_original);

  if(world_rank == 0 || world_rank == 2){
    MPI_Reduce(solution, global_solution, nUser_original, MPI_INT, MPI_SUM, 0, R_comm);
  }

  if(world_rank == 0){
    for(int i =0; i < nUser_original; i++)
      printf("%d\n", global_solution[i]);
  }

  //printf("complete\n");


  /******************************Free A*****************************/
  //printf("world_rank: %d nUser: %d\n", world_rank, nUser);
  /*for (int i = 0; i < nUser; i++) {
    
    A_aux1 = A_user[i];

    while (A_aux1 != NULL) {
      A_aux2 = A_aux1->nextItem;
      free(A_aux1);
      A_aux1 = A_aux2;
    }
  }*/
  //free(A_user);
  //free(A_item);
  /*****************************************************************/
  //free(solution);
  //printf("nuser total: %d nfeat: %d\n", nUser_original, nFeat);
  //free_LR(nUser_original, nFeat, &L, &R, &newL, &newR, &B);
  //free(local_B);
  //free(group_B);

  //printf("Finalizing\n");
  //elapsed_time += MPI_Wtime();
  printf("elapsed_time: %.1f\n", elapsed_time);

  MPI_Finalize();

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

entry** split_A(entry ***_A_user, entry ***_A_item, int new_first_user, int interval, int nU, int nI){

  entry **newU;
  entry *aux;
  entry *aux1, *aux2;
  int x=0;

  //percorrer o item para ter a certeza que todas as heads estão dentro da zona permitida

  for (int i = 0; i < nI; i++)
  {

  //printf("for %d done\n", i);
    if(((*_A_item)[i]) != NULL){
      /*
      printf("HERE %d\n", new_first_user);
      printf("user %d\n", ((*_A_item)[i])->user);
      printf("interval %d\n", interval);
      */
      if((((*_A_item)[i])->user)<new_first_user){
        //printf("HERE MACH\n");

        while((((*_A_item)[i])->user)<new_first_user){ //percorre até ultrapassar o valor do primeiro user

          //printf("HERE WH\n");
          (*_A_item)[i] = ((*_A_item)[i])->nextUser;
          if((*_A_item)[i] == NULL)
            break;
        }

        //printf("PRE IF\n");
        if((*_A_item)[i] != NULL) {
          if((((*_A_item)[i])->user)>=new_first_user+interval){ //ultrapassa o target
            /* meter a NULL */
            (*_A_item)[i] = NULL;
          }
        }


        }
      else if ((((*_A_item)[i])->user)>=new_first_user+interval){

        //printf("HERE NU\n");
        /* meter a NULL */
        (*_A_item)[i] = NULL;
      }
    }

    if((*_A_item)[i] != NULL) {
          //printf("HERE NE\n");
          aux1 = (*_A_item)[i];
          while(aux1->nextUser != NULL) {
            if(aux1->nextUser->user >= new_first_user+interval){
              aux1->nextUser = NULL;
            }
            else {
              aux1 = aux1->nextUser;
            }
          }
      }
      //else está dentro do target
  }


  //allocar um novo array para os users com a dimensao correcta
  newU = (entry **)calloc(sizeof(entry *), interval);
  
  
  //copiar parte do array total para o novo array

  for (int i = 0; i < nU; i++)
  {
    //if(((*_A_user)[i]) != NULL){

      if((i<new_first_user) || (i>=new_first_user+interval)){ //fora do target

        while(((*_A_user)[i])!=NULL){  //free das estruturas da lista

          aux = ((*_A_user)[i])->nextItem;
          free((*_A_user)[i]);
          (*_A_user)[i] = aux;
        }

      }
      else{ //dentro do target

        // meter a NULL
        newU[x]=(*_A_user)[i];
        /*
        if (x==interval-1)
        {
          aux = newU[x];
          while (aux!=NULL)
          {
            aux->nextUser=NULL;
            aux=aux->nextItem;
          }
        }
        */
        x++;
        (*_A_user)[i] = NULL;
      }     
   // }
  }


  

  //dar free do resto do array velho
  free(*_A_user);

  return newU;
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

void update_recom(int nU, int k0, int k1, double ***L, double ***R,
                 entry ***A_user, double *B, double *group_B, MPI_Comm group_comm, int sizeof_B) {
  entry *A_aux1;
  int k;
  //printf("HERE updating\n");
  // update recomendation for all non-zero entries meaning
  // the approximation of B to A
  for (int i = 0; i < nU; i++) {
    A_aux1 = (*A_user)[i];
    while (A_aux1 != NULL) {
      //printf("recom: %d\n", A_aux1->recom);
      B[A_aux1->recom] = 0; //B[A_aux->recom] = 0;
      group_B[A_aux1->recom] = 0;
      /*
      printf("recom 0\n");
      printf("k0: %d k1: %d\n", k0, k1);
      */

      for (k = k0; k < k1; k++){
        /*
        printf("k: %d\n", k);
        printf("user: %d item: %d\n", A_aux1->user, A_aux1->item);
        */
        B[A_aux1->recom] += (*L)[A_aux1->user][k] * (*R)[k][A_aux1->item]; //B[A_aux->recom] +=
      }

      //printf("going next\n");
      A_aux1 = A_aux1->nextItem;
    }
  }

  MPI_Allreduce(B, group_B, sizeof_B, MPI_DOUBLE, MPI_SUM, group_comm);
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

void update_LR(double ***L, double ***R, double ***newL, double ***newR) {

  double **aux;

  // update stable version of L with L(t+1) by switching
  // the pointers
  aux = *L;
  *L = *newL;
  *newL = aux;

  // update stable version of R with R(t+1) by switching
  // the pointers
  aux = *R;
  *R = *newR;
  *newR = aux;
}
