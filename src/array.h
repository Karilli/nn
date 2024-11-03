#ifndef ARRAY_H
#define ARRAY_H

#include "macro_utils.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "math.h"

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


void assert_vector_equals(Vector vec, FLOAT data[], int length, FLOAT eps) {
    ASSERT(vec.data != NULL, "Error in file %s, line %d.\n", __FILE__, __LINE__);
    ASSERT(length == vec.x_dim, "Expected length %d, but got %d.\n", length, vec.x_dim);
    for (int x=0; x < length; x++) {
        ASSERT(
            fabs(get_vector(vec, x) - data[x]) < eps,
            "Values at index %d are not equal. Expected %f, but got %f.\n",
            x, data[x], get_vector(vec, x)
        );
    }
}
void assert_matrix_equals(Matrix mat, FLOAT data[], int x_dim, int y_dim, FLOAT eps) {
    ASSERT(mat.data != NULL, "Error in file %s, line %d.\n", __FILE__, __LINE__);
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