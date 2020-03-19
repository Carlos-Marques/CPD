
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

    for(int i = 0; i < nEntry; i++){
        printf("%d %d %lf\n", A[i].column, A[i].row, A[i].rate);
    }

    multiply_LR(nUser, nItem, nFeat);
    for(int i = 0; i < nUser; i++){
        for(int j = 0; j < nFeat; j++)
            printf("%lf  ", L[i][j]);
        printf("\n");
    }
    printf("\n");

    for(int i = 0; i < nFeat; i++){
        for(int j = 0; j < nItem; j++)
            printf("%lf  ", R[i][j]);
        printf("\n");
    }
    printf("\n");

    for(int i = 0; i < nUser; i++){
        for(int j = 0; j < nItem; j++)
            printf("%lf  ", B[i][j]);
        printf("\n");
    }*/

    //printf("%d   %lf   %d   %d   %d   %d\n", nIter, alpha, nFeat, nUser, nItem,nZero);



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