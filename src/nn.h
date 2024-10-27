#ifndef NN_H
#define NN_H


#include "csv.h"
#include "array.h"
#include "assert.h"

#include <stdlib.h>
#include <time.h>
#include "math.h"


typedef struct Layer {
    Matrix parameters;
    Matrix gradients;
} Layer;


void init_layer(Layer *layer, int x_dim, int y_dim){
    init_matrix(&(layer->parameters), x_dim, y_dim);
    init_matrix(&(layer->gradients), x_dim, y_dim);
    srand(time(NULL));
    for(int y = 0; y < layer->parameters.y_dim; y++) {
        for(int x = 0; x < layer->parameters.x_dim; x++) {    
            set_matrix(layer->parameters, x, y, (float) rand() / RAND_MAX);
        }
    }
}


void delete_layer(Layer layer) {
    delete_matrix(layer.parameters);
    delete_matrix(layer.gradients);
}


Vector softmax(Vector vec) {
    Vector new;
    init_vector(&new, vec.x_dim);
    FLOAT sm = 0.0f;
    for (int x = 0; x < vec.x_dim; x++) {
        FLOAT val = exp(get_vector(vec, x));
        set_vector(new, x, val);
        sm += val;
    }
    for (int x = 0; x < vec.x_dim; x++) {
        set_vector(new, x, get_vector(new, x) / sm);
    }   
    delete_vector(vec);
    return new;
}


FLOAT cross_entropy(Vector vec, Vector target) {
    FLOAT error = 0;
    for (int x=0; x < vec.x_dim; x++) {
        error -= log(get_vector(vec, x)) * get_vector(target, x);
    }
    delete_vector(vec);
    return error;
}


Vector relu(Vector vec) {
    Vector new;
    init_vector(&new, vec.x_dim);

    for (int x = 0; x < vec.x_dim; x++) {
        FLOAT val = get_vector(vec, x);
        set_vector(new, x, (0 <= val) ? val : 0);
    } 

    delete_vector(vec);
    return new;
}


Vector matmul(Matrix mat, Vector vec) {
    Vector new;
    init_vector(&new, mat.y_dim);
    ASSERT(vec.x_dim == mat.x_dim,
        "Non-matching matrix sizes %d x %d, %d",
        mat.x_dim, mat.y_dim, vec.x_dim
    );
    for (int y=0; y < mat.y_dim; y++) {
        FLOAT sm = 0.0f;
        for (int x=0; x < mat.x_dim; x++) {
            sm += get_matrix(mat, x, y) * get_vector(vec, x);
        }
        set_vector(new, y, sm);
    }
    delete_vector(vec);
    return new;
}


#endif