#ifndef VECTOR_H
#define VECTOR_H

#include "macro_utils.h"
#include "precision.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


typedef struct Vector {
    int x_dim;
    FLOAT *data;
} Vector;


void init_vector(Vector *vec, int x_dim) {
    vec->x_dim = x_dim;
    MALLOC(vec->data, FLOAT, x_dim);
    memset(vec->data, 0.0, x_dim * sizeof(FLOAT));
}


void delete_vector(Vector vec) {
    FREE(vec.data);
}


FLOAT get_vector(Vector vec, int x) {
    ASSERT(vec.data != NULL, "Error in file %s, line %d.\n", __FILE__, __LINE__);
    ASSERT(
        0 <= x && x < vec.x_dim,
        "Vector index out of bounds x: 0 <= %d < %d.\n",
        x, vec.x_dim
    );
    return vec.data[x];  
}


void set_vector(Vector vec, int x, FLOAT value) {
    ASSERT(vec.data != NULL, "Error in file %s, line %d.\n", __FILE__, __LINE__);
    ASSERT(
        0 <= x && x < vec.x_dim,
        "Vector index out of bounds x: 0 <= %d < %d.\n",
        x, vec.x_dim
    );
    vec.data[x] = value;  
}


void add_vector(Vector vec, int x, FLOAT value) {
    ASSERT(vec.data != NULL, "Error in file %s, line %d.\n", __FILE__, __LINE__);
    ASSERT(
        0 <= x && x < vec.x_dim,
        "Vector index out of bounds x: 0 <= %d < %d.\n",
        x, vec.x_dim
    );
    vec.data[x] += value;  
}


void print_vector(Vector vec) {
    ASSERT(vec.data != NULL, "Error in file %s, line %d.\n", __FILE__, __LINE__);
    for(int i = 0; i < vec.x_dim; i++) {
        printf("%f, ", vec.data[i]);
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


#endif