#ifndef ARRAY_H
#define ARRAY_H

#include "macro_utils.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include <stdatomic.h>
#include <float.h>

#define FLOAT float


typedef struct Matrix {
    int x_dim;
    int y_dim;
    FLOAT *data;
} Matrix;


typedef struct Vector {
    int x_dim;
    FLOAT *data;
} Vector;


void init_matrix(Matrix *mat, int x_dim, int y_dim) {
    mat->x_dim = x_dim;
    mat->y_dim = y_dim;
    MALLOC(mat->data, FLOAT, (unsigned int) (x_dim * y_dim));
    memset(mat->data, 0.0f, (unsigned int) (x_dim * y_dim) * sizeof(FLOAT));
}


void init_vector(Vector *vec, int x_dim) {
    vec->x_dim = x_dim;
    MALLOC(vec->data, FLOAT, (unsigned int) x_dim);
    memset(vec->data, 0.0f, (unsigned int) x_dim * sizeof(FLOAT));
}


FLOAT get_matrix(Matrix mat, int x, int y) {
    ASSERT(
        0 <= x && x < mat.x_dim && 0 <= y && y < mat.y_dim,
        "Matrix indices out of bounds x: 0 <= %d < %d, y: 0 <= %d < %d.",
        x, mat.x_dim, y, mat.y_dim
    );
    return mat.data[y * mat.x_dim + x];  
}


FLOAT get_vector(Vector vec, int x) {
    ASSERT(
        0 <= x && x < vec.x_dim,
        "Vector index out of bounds x: 0 <= %d < %d.\n",
        x, vec.x_dim
    );
    return vec.data[x];  
}


void set_matrix(Matrix mat, int x, int y, FLOAT value) {
    ASSERT(
        0 <= x && x < mat.x_dim && 0 <= y && y < mat.y_dim,
        "Matrix indices out of bounds x: 0 <= %d < %d, y: 0 <= %d < %d.",
        x, mat.x_dim, y, mat.y_dim
    );
    mat.data[y * mat.x_dim + x] = value;  
}


void add_matrix(Matrix mat, int x, int y, FLOAT value) {
    ASSERT(
        0 <= x && x < mat.x_dim && 0 <= y && y < mat.y_dim,
        "Matrix indices out of bounds x: 0 <= %d < %d, y: 0 <= %d < %d.",
        x, mat.x_dim, y, mat.y_dim
    );
    mat.data[y * mat.x_dim + x] += value;  
}


void set_vector(Vector vec, int x, FLOAT value) {
    ASSERT(
        0 <= x && x < vec.x_dim,
        "Vector index out of bounds x: 0 <= %d < %d.\n",
        x, vec.x_dim
    );
    vec.data[x] = value;  
}


void add_vector(Vector vec, int x, FLOAT value) {
    ASSERT(
        0 <= x && x < vec.x_dim,
        "Vector index out of bounds x: 0 <= %d < %d.\n",
        x, vec.x_dim
    );
    vec.data[x] += value;  
}


void delete_vector(Vector vec) {
    FREE(vec.data);
}


void delete_matrix(Matrix mat) {
    FREE(mat.data);
}


void print_vector(Vector vec) {
    for(int i = 0; i < vec.x_dim; i++) {
        printf("%f, ", vec.data[i]);
    }
    printf("\n");
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


FLOAT matrix_max(Matrix mat) {
    FLOAT res = -FLT_MAX;

    for (int y = 0; y < mat.y_dim; y++) {
        for(int x = 0; x < mat.x_dim; x++) {
            FLOAT val = get_matrix(mat, x, y);
            if (res < val) {
                res = val;
            }
        }
    }
    return res;
}


#endif