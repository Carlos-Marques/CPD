/**************************Declarations**************************/

#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef struct entryA{
    int user;
    int item;
    double rate;
    struct entryA* nextItem;
    struct entryA* nextUser;
}entryA;

double **L, **R, **B, **newL, **newR;

entryA* createNode();

void alloc_LR(int nU, int nI, int nF, double ***L, double ***R, double ***newL, double ***newR, double ***B);
void random_fill_LR(int nU, int nI, int nF, double ***L, double ***R, double ***newL, double ***newR);
void multiply_LR(int nU, int nI, int nF, double ***L, double ***R, double ***B);
void update_LR(int nU, int nI, int nF, double ***L, double ***R, double ***newL, double ***newR);
void free_LR(int nU, int nF, double ***L, double ***R, double ***newL, double ***newR, double ***B);

/****************************************************************/

int main(int argc, char *argv[]){
    FILE *fp;
    int nIter, nFeat, nUser, nItem, nZero, nEntry;
    double deriv = 0;
    double alpha;
    double **L, **R, **B, **newL, **newR;

    entryA **A_user, **A_user_aux, **A_item, **A_item_aux;
    entryA *A_aux1, *A_aux2;

    if(argc != 2){
		printf("error: command of type ./matFact <filename.in>\n");
		exit(1);
	}

    fp = fopen(argv[1], "r");
    if(fp == NULL){
        printf("error: cannot open file\n");
		exit(1);
    }

/******************************Setup******************************/
    //Functional Parallelism
    fscanf(fp, "%d", &nIter);
    fscanf(fp, "%lf", &alpha);
    fscanf(fp, "%d", &nFeat);
    fscanf(fp, "%d %d %d", &nUser, &nItem, &nEntry);

    A_user = (entryA**)calloc(sizeof(entryA*), nUser);
    A_item = (entryA**)calloc(sizeof(entryA*), nItem);

    A_user_aux = (entryA**)calloc(sizeof(entryA*), nUser);
    A_item_aux = (entryA**)calloc(sizeof(entryA*), nItem);

    //Data Parallelism
    for(int i = 0; i < nEntry; i++){
        
        A_aux1 = createNode();
        
        fscanf(fp, "%d %d %lf", &(A_aux1->user), &(A_aux1->item), &(A_aux1->rate));

        if(A_user[A_aux1->user] == NULL){

            A_user[A_aux1->user] = A_aux1;
            A_user_aux[A_aux1->user] = A_aux1;

        }else{

            A_user_aux[A_aux1->user]->nextItem = A_aux1;
            A_user_aux[A_aux1->user] = A_aux1;
        }



        if(A_item[A_aux1->item] == NULL){

            A_item[A_aux1->item] = A_aux1;
            A_item_aux[A_aux1->item] = A_aux1;

        }else{

            A_item_aux[A_aux1->item]->nextUser = A_aux1;
            A_item_aux[A_aux1->item] = A_aux1;
        }
    }

    fclose(fp);
    free(A_item_aux);
    free(A_user_aux);

    alloc_LR(nUser, nItem, nFeat, &L, &R, &newL, &newR, &B);
    random_fill_LR(nUser, nItem, nFeat, &L, &R, &newL, &newR);
    multiply_LR(nUser, nItem, nFeat, &L, &R, &B);    

    /****************************End Setup****************************/

    /***********************Matrix Factorization**********************/
    for(int n = 0; n < nIter; n++){

        
        //Matrix L
        for(int i = 0; i < nUser; i++){
            for(int k = 0; k < nFeat; k++){

                A_aux1 = A_user[i];
                while(A_aux1 != NULL){
                    deriv += 2*(A_aux1->rate - B[i][A_aux1->item])*(-R[k][A_aux1->item]);
                    A_aux1 = A_aux1->nextItem;
                }

                newL[i][k] = L[i][k]-alpha*deriv;
                deriv = 0;
            }
        }

        //Matrix R
        for(int k = 0; k < nFeat; k++){
            for(int j = 0; j < nItem; j++){

                A_aux1 = A_item[j];
                while(A_aux1 != NULL){
                    deriv += 2*(A_aux1->rate - B[A_aux1->user][j])*(-L[A_aux1->user][k]);
                    A_aux1 = A_aux1->nextUser;
                }

                newR[k][j] = R[k][j]-alpha*deriv;
                deriv = 0;
            }
        }
  
        update_LR(nUser, nItem, nFeat, &L, &R, &newL, &newR);   
        multiply_LR(nUser, nItem, nFeat, &L, &R, &B);   
    }
    /*********************End Matrix Factorization********************/

    for(int i = 0; i < nUser; i++){
        for(int j = 0; j < nItem; j++)
            printf("%lf  ", B[i][j]);
        printf("\n");
    }    

    /******************************Free A*****************************/
    for(int i = 0; i < nUser; i++){

        A_aux1 = A_user[i];

        while(A_aux1 != NULL){
            A_aux2 = A_aux1->nextItem;
            free(A_aux1);
            A_aux1 = A_aux2;
        }
        
    }
    free(A_user);
    free(A_item);
    /*****************************************************************/

    free_LR(nUser, nFeat, &L, &R, &newL, &newR, &B);

    return 0;
}

entryA* createNode(){

    entryA *A;
    A = (entryA*)malloc(sizeof(entryA));
    A->nextItem = NULL;
    A->
    nextUser = NULL;

    return A;
}


void alloc_LR(int nU, int nI, int nF, double ***L, double ***R, double ***newL, double ***newR, double ***B){

    //Functional Parallelism
    *B = (double**)malloc(sizeof(double*)*nU);
    *L = (double**)malloc(sizeof(double*)*nU);
    *newL = (double**)malloc(sizeof(double*)*nU);
    *R = (double**)malloc(sizeof(double*)*nF);
    *newR = (double**)malloc(sizeof(double*)*nF);

    //Data Parallelism
	for (int i = 0; i < nU; i++){
        (*B)[i] = (double*)malloc(sizeof(double)*nI);
        (*L)[i] = (double *)malloc(sizeof(double) *nF);
        (*newL)[i] = (double *)malloc(sizeof(double)*nF);
    }

    //Data Parallelism
    for (int i = 0; i < nF; i++){
	    (*R)[i] = (double *)malloc(sizeof(double)*nI);
        (*newR)[i] = (double *)malloc(sizeof(double)*nI);
    }	
}


void random_fill_LR(int nU, int nI, int nF, double ***L, double ***R, double ***newL, double ***newR)
{
    srandom(0);
    
    //Data Parallelism
    for(int i = 0; i < nU; i++)
        for(int j = 0; j < nF; j++){
            (*L)[i][j] = RAND01 / (double) nF;
            (*newL)[i][j] = (*L)[i][j]; 
        }
    
    //Data Parallelism
    for(int i = 0; i < nF; i++)
        for(int j = 0; j < nI; j++){
            (*R)[i][j] = RAND01 / (double) nF;
            (*newR)[i][j] = (*R)[i][j];
        }
}   


void multiply_LR(int nU, int nI, int nF, double ***L, double ***R, double ***B){
    
    for(int i = 0; i < nU; i++)
        for(int j = 0; j < nI; j++)
            (*B)[i][j] = 0;

    //Data Parallelism
    for(int i = 0; i < nU; i++)
        for(int j = 0; j < nI; j++)
            for(int k = 0; k < nF; k++)
                (*B)[i][j] += (*L)[i][k]*(*R)[k][j];
}


void update_LR(int nU, int nI, int nF, double ***L, double ***R, double ***newL, double ***newR){

    double **aux;
    aux = *L;
    *L = *newL;
    *newL = aux;

    aux = *R;
    *R = *newR;
    *newR = aux;
}

void free_LR(int nU, int nF, double ***L, double ***R, double ***newL, double ***newR, double ***B){

    //Data Parallelism
	for(int i=0; i<nU; i++){
        free((*B)[i]);
        free((*L)[i]);
        free((*newL)[i]);
    } 
	free(*B);
	free(*L);
	free(*newL);

    //Data Parallelism
    for(int i=0; i<nF; i++){
        free((*R)[i]);
        free((*newR)[i]);
    } 
	free(*newR);
	free(*R);
}
