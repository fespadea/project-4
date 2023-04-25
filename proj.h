#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double normFro2(double** A, int n, int m){
    double sum = 0;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            sum += A[i][j] * A[i][j];
        }
    }
    double totalSum = 0;
    MPI_Reduce(&sum, &totalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&totalSum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return totalSum;
}

double norm1(double** A, int n, int m, int rank){
    double sumMax = 0;
    for(int j = 0; j < m; j++){
        double sum = 0;
        for(int i = 0; i < n; i++){
            sum += fabs(A[i][j]);
        }
        double totalSum = 0;
        MPI_Reduce(&sum, &totalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(rank == 0){
            if(totalSum > sumMax){
                sumMax = totalSum;
            }
        }
    }
    MPI_Bcast(&sumMax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return sumMax;
}

// Code from geeksforgeeks: https://www.geeksforgeeks.org/program-for-rank-of-matrix/
void swap(double ** mat, int R, int C, int row1, int row2,
          int col)
{
    for (int i = 0; i < col; i++)
    {
        int temp = mat[row1][i];
        mat[row1][i] = mat[row2][i];
        mat[row2][i] = temp;
    }
}

// Code from geeksforgeeks: https://www.geeksforgeeks.org/program-for-rank-of-matrix/
int rankOfMatrix(double ** mat, int R, int C)
{
    double ** matCopy = (double **)malloc(R*sizeof(double*));
    for(int i = 0; i < R; i++){
        matCopy[i] = (double *)malloc(C*sizeof(double));
        for(int j = 0; j < C; j++){
            matCopy[i][j] = mat[i][j];
        }
    }
    mat = matCopy;

    int rank = C;
 
    for (int row = 0; row < rank; row++)
    {
        // Before we visit current row 'row', we make
        // sure that mat[row][0],....mat[row][row-1]
        // are 0.
 
        // Diagonal element is not zero
        if (mat[row][row])
        {
           for (int col = 0; col < R; col++)
           {
               if (col != row)
               {
                 // This makes all entries of current
                 // column as 0 except entry 'mat[row][row]'
                 double mult = (double)mat[col][row] /
                                       mat[row][row];
                 for (int i = 0; i < rank; i++)
                   mat[col][i] -= mult * mat[row][i];
              }
           }
        }
 
        // Diagonal element is already zero. Two cases
        // arise:
        // 1) If there is a row below it with non-zero
        //    entry, then swap this row with that row
        //    and process that row
        // 2) If all elements in current column below
        //    mat[r][row] are 0, then remove this column
        //    by swapping it with last column and
        //    reducing number of columns by 1.
        else
        {
            int reduce = 1;
 
            /* Find the non-zero element in current
                column  */
            for (int i = row + 1; i < R;  i++)
            {
                // Swap the row with non-zero element
                // with this row.
                if (mat[i][row])
                {
                    swap(mat, R, C, row, i, rank);
                    reduce = 0;
                    break;
                }
            }
 
            // If we did not find any row with non-zero
            // element in current column, then all
            // values in this column are 0.
            if (reduce)
            {
                // Reduce number of columns
                rank--;
 
                // Copy the last column here
                for (int i = 0; i < R; i ++)
                    mat[i][row] = mat[i][rank];
            }
 
            // Process this row again
            row--;
        }
    }
    free(mat);
    return rank;
}


int cmp (const void * a, const void * b) {
    double doubleA = *((double *) a);
    double doubleB = *((double *) b);
    if (doubleA == doubleB) return 0;
    else if (doubleA < doubleB) return -1;
    else return 1;
}

double ** matrixSparsification(double ** A, int n, int m, double epsilon, double delta, int sMult, double alpha, int totaln, int rank, int nranks){
    // get paramters ready
    double AF2 = normFro2(A, n, m);
    double A1 = norm1(A, n, m, rank);
    int k;
    if(rank == 0){
        k = rankOfMatrix(A, n, m);
    }
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int s = sMult * k * (totaln + m);

    // calculate probabilities for each value of being chosen
    int totalLength = n*m;
    double * probabilities = (double *)malloc(sizeof(double)*(totalLength * (rank == 0 ? nranks : 1)));
    double sum = 0;
    for(int i = 0; i < totalLength; i++){
        double Aij = A[(int)(i / m)][i % m];
        probabilities[i] = alpha * fabs(Aij) / A1 + (1 - alpha) * (Aij * Aij) / AF2;
        sum += probabilities[i];
    }
    double totalSum = 0;
    MPI_Reduce(&sum, &totalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&totalSum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    sum = totalSum;


    int * choices = (int *)malloc(sizeof(int)*s);
    if(rank != 0){
        MPI_Request request;
        MPI_Isend(probabilities, totalLength, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);
        MPI_Status status;
        MPI_Waitall(1, &request, &status);

        int i = 0;
        do{
            MPI_Irecv(choices + i, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &request);
            MPI_Waitall(1, &request, &status);
        } while(choices[i++] != -1);
    } else{
        MPI_Request * requests = (MPI_Request *)malloc(sizeof(MPI_Request)*(nranks-1));
        for(int i = 1; i < nranks; i++){
            MPI_Irecv(probabilities + i*totalLength, totalLength, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &requests[i-1]);
        }

        // choose the indexes using the probabilities
        double * probs = (double *)malloc(sizeof(double)*s);\
        for(int i = 0; i < s; i++){
            probs[i] = ((double)rand() / (double)RAND_MAX) * sum;
        }
        qsort(probs, s, sizeof(double), cmp);
        double probSum = 0;
        int p = 0;
        MPI_Request request;
        MPI_Status status;
        MPI_Status  * statuses = (MPI_Status  *)malloc(sizeof(MPI_Status )*(nranks-1));
        MPI_Waitall(nranks-1, requests, statuses);
        free(requests);
        free(statuses);
        for(int j = 0; p < s; j++){
            probSum += probabilities[j];
            while(probSum >= probs[p] && p < s){
                if(j < totalLength){
                    choices[p] = j;
                } else{
                    choices[p] = -2;
                    int choice = j % totalLength;
                    MPI_Isend(&choice, 1, MPI_INT, j/totalLength, 1, MPI_COMM_WORLD, &request);
                    MPI_Waitall(1, &request, &status);
                }
                p++;
            }
        }

        for(int i = 1; i < nranks; i++){
            int done = -1;
            MPI_Send(&done, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }

        free(probs);
    }


    // combine the chosen values into a sparse matrix
    double ** ATilde = (double**)calloc(n * sizeof(double*), sizeof(double*));
    for(int i = 0; i < n; i++){
        ATilde[i] = (double*)calloc(m * sizeof(double), sizeof(double*));
    }
    for(int k = 0; k < s; k++){
        if(choices[k] >= 0){
            int choice = choices[k];
            int i = (int)(choice / m);
            int j = choice % m;
            ATilde[i][j] += A[i][j] / probabilities[choice] / s;
        } else if(choices[k] == -1){
            break;
        }
    }
    
    free(probabilities);
    free(choices);

    return ATilde;
}

double error(double ** A, double ** ATilde, int n, int m, int rank){
    double ** ADiff = (double**)malloc(n * sizeof(double*));
    for(int i = 0; i < n; i++){
        ADiff[i] = (double*)malloc(m * sizeof(double));
        for(int j = 0; j < m; j++){
            ADiff[i][j] = A[i][j] - ATilde[i][j];
        }
    }
    double error = norm1(ADiff, n, m, rank) / norm1(A, n, m, rank);
    free(ADiff);
    return log(error) / log(2);
}