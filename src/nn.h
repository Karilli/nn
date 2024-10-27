#ifndef NN_H
#define NN_H


#include "csv.h"
#include "array.h"
#include "assert.h"

#include <stdlib.h>
#include <time.h>
#include "math.h"


typedef struct Parameters {
    Matrix parameters;
    Matrix gradients;
} Parameters;


void init_params(Parameters *params, int x_dim, int y_dim){
    init_matrix(&(params->parameters), x_dim, y_dim);
    init_matrix(&(params->gradients), x_dim, y_dim);
    srand(time(NULL));
    for(int y = 0; y < params->parameters.y_dim; y++) {
        for(int x = 0; x < params->parameters.x_dim; x++) {    
            set_matrix(params->parameters, x, y, (float) rand() / RAND_MAX);
        }
    }
}


void delete_params(Parameters parameters) {
    delete_matrix(parameters.parameters);
    delete_matrix(parameters.gradients);
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


Vector matmul(Parameters parameters, Vector vec) {
    Vector new;
    init_vector(&new, parameters.parameters.y_dim);
    ASSERT(vec.x_dim == parameters.parameters.x_dim,
        "Non-matching matrix sizes %d x %d, %d",
        parameters.parameters.x_dim, parameters.parameters.y_dim, vec.x_dim
    );
    for (int y=0; y < parameters.parameters.y_dim; y++) {
        FLOAT sm = 0.0f;
        for (int x=0; x < parameters.parameters.x_dim; x++) {
            sm += get_matrix(parameters.parameters, x, y) * get_vector(vec, x);
        }
        set_vector(new, y, sm);
    }
    delete_vector(vec);
    return new;
}


#endif