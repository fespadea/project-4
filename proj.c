#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "clockcycle.h"
#include "proj.h"

#define clock_frequency 512000000

int main(int argc, char **argv){
    FILE* metaDataFile = fopen("formattedMetaData.tsv", "r");
    int n;
    fscanf(metaDataFile, "%d", &n);
    n /= atoi(argv[2]);
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

    unsigned long long start_cycles= clock_now();
    
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

    double epsilon = 0.05;
    double delta = 0.1;
    int sMult = atoi(argv[1]);
    double alpha = 0.5;

    double ** ATilde = matrixSparsification(dataMatrix, localn, m, epsilon, delta, sMult, alpha, n, rank, nprocs);

    int buf2size = totalLength*sizeof(double)/nprocs;
    double *buf2 = (double *)malloc(buf2size);
    int ndoubles = buf2size/sizeof(double);

    for(int i = 0; i < localn; i++){
        for(int j = 0; j < m; j++){
            buf2[i*m + j] = ATilde[i][j];
        }
    }
    MPI_File_open(MPI_COMM_WORLD, "output.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at(fh, rank*buf2size, buf2, ndoubles, MPI_DOUBLE, &status);
    MPI_File_close(&fh);

    unsigned long long end_cycles= clock_now();

    double errorResult = error(dataMatrix, ATilde, localn, m, rank);
    if(rank == 0){
        printf("Error: %lf\n", errorResult);
        double time_in_secs_CUDA = ((double)(end_cycles - start_cycles)) / clock_frequency;
        printf("CUDA Reduce Sum Seconds Taken: %lf\n", time_in_secs_CUDA);
    }

    free(ATilde);
    free(dataMatrix);
    free(buf);


    MPI_Finalize();
}