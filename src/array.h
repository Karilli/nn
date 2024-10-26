#ifndef ARRAY_H
#define ARRAY_H

#include "assert.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define FLOAT double


typedef struct Matrix {
    int x_dim;
    int y_dim;
    FLOAT *data;
} Matrix;


typedef struct Vector {
    int x_dim;
    FLOAT *data;
} Vector;


void print_vector(Vector vec) {
    for(int i = 0; i < vec.x_dim; i++) {
        printf("%f, ", vec.data[i]);
    }
    printf("\n");
}


void init_matrix(Matrix *mat, int x_dim, int y_dim) {
    mat->x_dim = x_dim;
    mat->y_dim = y_dim;
    mat->data = (FLOAT *)malloc(x_dim * y_dim * sizeof(FLOAT));
    memset(mat->data, 0.0f, x_dim * y_dim * sizeof(FLOAT));
    ASSERT(
        mat->data != NULL, 
        "Memory allocation failed for Matrix"
    );
}


void init_vector(Vector *vec, int length) {
    vec->x_dim = length;
    vec->data = (FLOAT *)malloc(length * sizeof(FLOAT));
    ASSERT(
        vec->data != NULL,
        "Memory allocation failed for Vector"
    );
}


FLOAT get_matrix(Matrix mat, int x, int y) {
    ASSERT(
        0 <= x && x < mat.x_dim && 0 <= y && y < mat.y_dim,
        "Matrix indices out of bounds %d, %d %d, %d",
        x, mat.x_dim, y, mat.y_dim
    );
    return mat.data[y * mat.x_dim + x];  
}


FLOAT get_vector(Vector vec, int x) {
    ASSERT(
        0 <= x && x < vec.x_dim,
        "Vector index out of bounds."
    );
    return vec.data[x];  
}


void set_matrix(Matrix *mat, int x, int y, FLOAT value) {
    ASSERT(
        0 <= x && x < mat->x_dim && 0 <= y && y < mat->y_dim,
        "Matrix indices out of bounds %d, %d %d, %d",
        x, mat->x_dim, y, mat->y_dim
    );
    mat->data[y * mat->x_dim + x] = value;  
}


void set_vector(Vector *vec, int x, FLOAT value) {
    ASSERT(
        0 <= x && x < vec->x_dim,
        "Vector index out of bounds."
    );    vec->data[x] = value;  
}


void delete_vector(Vector vec) {
    free(vec.data);
}


#endif