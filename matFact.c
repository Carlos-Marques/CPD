/**************************Declarations***********************************/

#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

double **L, **R;

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef struct entryA{
    int row;
    int column;
    double rate;
}entryA;

void random_fill_LR(int nU, int nI, int nF);
void alloc_LR(int nU, int nI, int nF);
void free_LR(int nU, int nF);

/***********************************************************************/


int main(int argc, char *argv[]){
    FILE *fp;
    int nIter, nFeat, nUser, nItem, nZero, noEntry;
    double alpha;

    if(argc != 2){
		printf("error: command of type ./matFact <filename.in>\n");
		exit(1);
	}

    fp = fopen(argv[1], "r");
    if(fp == NULL){
        printf("error: cannot open file\n");
		exit(1);
    }

    //Functional Parallelism
    fscanf(fp, "%d", &nIter);
    fscanf(fp, "%lf", &alpha);
    fscanf(fp, "%d", &nFeat);
    fscanf(fp, "%d %d %d", &nUser, &nItem, &noEntry);

    entryA* A = (entryA *)malloc(sizeof(entryA)*noEntry);

    //Data Parallelism
    for(int i = 0; i < noEntry; i++){
        fscanf(fp, "%d %d %lf", &(A[i].column), &(A[i].row), &(A[i].rate));
    }

    alloc_LR(nUser, nItem, nFeat);
    random_fill_LR(nUser, nItem, nFeat);

    

    fclose(fp);
    free(A);
    free_LR(nUser, nFeat);

    return 0;
}

void alloc_LR(int nU, int nI, int nF){

    L = (double**)malloc(sizeof(double)*nU);
    R = (double**)malloc(sizeof(double)*nF);

    //Data Parallelism
	for (int i = 0; i < nU; i++)
		L[i] = (double *)malloc(sizeof(double)  * nF);
    //Data Parallelism
    for (int i = 0; i < nF; i++)
		R[i] = (double *)malloc(sizeof(double) * nI);
}


void random_fill_LR(int nU, int nI, int nF)
{
    srandom(0);
    
    //Data Parallelism
    for(int i = 0; i < nU; i++)
        for(int j = 0; j < nF; j++)
            L[i][j] = RAND01 / (double) nF;
    
    //Data Parallelism
    for(int i = 0; i < nF; i++)
        for(int j = 0; j < nI; j++)
            R[i][j] = RAND01 / (double) nF;
}

void free_LR(int nU, int nF){

    //Data Parallelism
	for(int i=0; i<nU; i++) free(L[i]);
	free(L);

    //Data Parallelism
    for(int i=0; i<nF; i++) free(R[i]);
	free(R);
}


/*main(int argc, char *argv[]){
    int fd;
    char *filename;

    if(argc != 2){
		printf("error: command of type matFact <filename.in>\n");
		exit(1);
	}
    else{ strcpy(filename, argv[1]); }

    fd = fileno(filename);
}*/


/*for(int i = 0; i < nUser; i++)
        for(int j = 0; j < nFeat; j++)
            printf("%lf    \n", L[i][j]);
    for(int i = 0; i < nFeat; i++)
        for(int j = 0; j < nItem; j++)
            printf("%lf    \n", R[i][j]);

    for(int i = 0; i < noEntry; i++){
        printf("%d %d %lf\n", A[i].column, A[i].row, A[i].rate);
    }*/

    //printf("%d   %lf   %d   %d   %d   %d\n", nIter, alpha, nFeat, nUser, nItem,nZero);