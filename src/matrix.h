#ifndef MATRIX_H
#define MATRIX_H


#include "macro_utils.h"
#include "precision.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


typedef struct Matrix {
    int x_dim;
    int y_dim;
    FLOAT *data;
} Matrix;


void init_matrix(Matrix *mat, int x_dim, int y_dim) {
    mat->x_dim = x_dim;
    mat->y_dim = y_dim;
    MALLOC(mat->data, FLOAT, x_dim * y_dim);
    memset(mat->data, 0.0, x_dim * y_dim * sizeof(FLOAT));
}


void delete_matrix(Matrix mat) {
    FREE(mat.data);
}


FLOAT get_matrix(Matrix mat, int x, int y) {
    ASSERT(
        0 <= x && x < mat.x_dim && 0 <= y && y < mat.y_dim,
        "Matrix indices out of bounds x: 0 <= %d < %d, y: 0 <= %d < %d.\n",
        x, mat.x_dim, y, mat.y_dim
    );
    return mat.data[y * mat.x_dim + x];  
}


void set_matrix(Matrix mat, int x, int y, FLOAT value) {
    ASSERT(
        0 <= x && x < mat.x_dim && 0 <= y && y < mat.y_dim,
        "Matrix indices out of bounds x: 0 <= %d < %d, y: 0 <= %d < %d.\n",
        x, mat.x_dim, y, mat.y_dim
    );
    mat.data[y * mat.x_dim + x] = value;  
}


void add_matrix(Matrix mat, int x, int y, FLOAT value) {
    ASSERT(
        0 <= x && x < mat.x_dim && 0 <= y && y < mat.y_dim,
        "Matrix indices out of bounds x: 0 <= %d < %d, y: 0 <= %d < %d.\n",
        x, mat.x_dim, y, mat.y_dim
    );
    mat.data[y * mat.x_dim + x] += value;  
}


void print_matrix(Matrix mat) {
    for (int y = 0; y < mat.y_dim; y++) {
        for(int x = 0; x < mat.x_dim; x++) {
            printf("%f, ", get_matrix(mat, x, y));
        }
        printf("\n");
    }
    printf("\n");
}


void assert_matrix_equals(Matrix mat, FLOAT data[], int x_dim, int y_dim, FLOAT eps) {
    ASSERT(x_dim == mat.x_dim, "Expected x_dim %d, but got %d.\n", x_dim, mat.x_dim);
    ASSERT(y_dim == mat.y_dim, "Expected y_dim %d, but got %d.\n", y_dim, mat.y_dim);
    for (int y=0; y < y_dim; y++) {
        for (int x=0; x < x_dim; x++) {
            ASSERT(
                fabs(get_matrix(mat, x, y) - data[y * x_dim + x]) < eps,
                "Values at %d,%d are not equal. Expected %f, but got %f.\n",
                x, y, data[y * x_dim + x], get_matrix(mat, x, y)
            );
        }
    }
}


#endif