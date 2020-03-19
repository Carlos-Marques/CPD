/**************************Declarations**************************/

#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <fcntl.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef struct entryA{
    int row;
    int column;
    double rate;
}entryA;

void random_fill_LR(int nU, int nI, int nF, double ***L, double ***R, double ***newL, double ***newR);
void alloc_LR(int nU, int nI, int nF, double ***L, double ***R, double ***newL, double ***newR, double ***B);
void free_LR(int nU, int nF, double ***L, double ***R, double ***newL, double ***newR, double ***B);
void update_LR(int nU, int nI, int nF, double ***L, double ***R, double ***newL, double ***newR);
void multiply_LR(int nU, int nI, int nF, double ***L, double ***R, double ***B);

/****************************************************************/

int main(int argc, char *argv[]){
    FILE *fp;
    int nIter, nFeat, nUser, nItem, nZero, nEntry;
    double deriv = 0;
    double alpha;
    double **L, **R, **B, **newL, **newR;
    entryA* A;

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

    A = (entryA*)malloc(sizeof(entryA)*nEntry);

    //Data Parallelism
    for(int i = 0; i < nEntry; i++){
        fscanf(fp, "%d %d %lf", &(A[i].row), &(A[i].column), &(A[i].rate));
    }

    fclose(fp);

    alloc_LR(nUser, nItem, nFeat, &L, &R, &newL, &newR, &B);
    random_fill_LR(nUser, nItem, nFeat, &L, &R, &newL, &newR);
    multiply_LR(nUser, nItem, nFeat, &L, &R, &B);    

/****************************************************************/
 
    for(int n = 0; n < nIter; n++){

        for(int i = 0; i < nUser; i++){
            for(int k = 0; k < nFeat; k++){

                for(int m = 0; m < nEntry; m++){
                    if(A[m].row == i){
                        deriv += 2*(A[m].rate - B[i][A[m].column])*(-R[k][A[m].column]);
                    }
                }
                newL[i][k] = L[i][k]-alpha*deriv;
                deriv = 0;
            }
        }

        for(int k = 0; k < nFeat; k++){
            for(int j = 0; j < nItem; j++){

                for(int m = 0; m < nEntry; m++){
                    if(A[m].column == j){
                        deriv += 2*(A[m].rate - B[A[m].row][j])*(-L[A[m].row][k]);
                    }
                }
                newR[k][j] = R[k][j]-alpha*deriv;
                deriv = 0;
            }
        }
  
        update_LR(nUser, nItem, nFeat, &L, &R, &newL, &newR);   
        multiply_LR(nUser, nItem, nFeat, &L, &R, &B);   
    }

    for(int i = 0; i < nUser; i++){
        for(int j = 0; j < nItem; j++)
            printf("%lf  ", B[i][j]);
        printf("\n");
    }    

/****************************************************************/

    free(A);
    free_LR(nUser, nFeat, &L, &R, &newL, &newR, &B);

    return 0;
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

void update_LR(int nU, int nI, int nF, double ***L, double ***R, double ***newL, double ***newR){

    double **aux;
    aux = *L;
    *L = *newL;
    *newL = aux;

    aux = *R;
    *R = *newR;
    *newR = aux;
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