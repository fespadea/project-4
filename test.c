#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "clockcycle.h"
#include "proj.h"

#define clock_frequency 512000000

int main(int argc, char **argv){
    FILE* metaDataFile = fopen("formattedMetaData.dat", "r");
    int n;
    fscanf(metaDataFile, "%d", &n);
    int m;
    fscanf(metaDataFile, "%d", &m);
    fclose(metaDataFile);
    // printf("%d %d\n", n, m);
    int totalLength = n * m;

    
    int rank, nprocs;
    MPI_File fh;
    MPI_Status status;
    int bufsize, nints;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    bufsize = totalLength*sizeof(int)/nprocs;
    int *buf = (int *)malloc(bufsize);
    nints = bufsize/sizeof(int);

    
    MPI_File_open(MPI_COMM_WORLD, "formattedData.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, rank*bufsize, buf, nints, MPI_INT, &status);
    MPI_File_close(&fh);
    
    double ** dataMatrix = (double**)malloc(n * sizeof(double*));
    int localn = nints/m;
    for(int i = 0; i < localn; i++){
        dataMatrix[i] = (double*)malloc(m * sizeof(double));
        for(int j = 0; j < m; j++){
            dataMatrix[i][j] = buf[i*m + j];
        }
    }
    
    int buf2size = totalLength*sizeof(double)/nprocs;
    double *buf2 = (double *)malloc(buf2size);
    int ndoubles = buf2size/sizeof(double);

    MPI_File_open(MPI_COMM_WORLD, "output.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, rank*buf2size, buf2, ndoubles, MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    
    double ** ATilde = (double**)malloc(n * sizeof(double*));
    for(int i = 0; i < localn; i++){
        ATilde[i] = (double*)malloc(m * sizeof(double));
        for(int j = 0; j < m; j++){
            ATilde[i][j] = buf2[i*m + j];
        }
    }

    double errorResult = error(dataMatrix, ATilde, localn, m, rank);
    if(rank == 0){
        printf("Error: %lf\n", errorResult);
    }

    free(ATilde);
    free(dataMatrix);
    free(buf);
    free(buf2);


    MPI_Finalize();
}