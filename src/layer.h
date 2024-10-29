#ifndef LAYER_H
#define LAYER_H


#include "csv.h"
#include "vector.h"
#include "matrix.h"
#include "assert.h"

#include <stdlib.h>
#include <time.h>
#include "math.h"
#include "stdbool.h"


typedef struct Layer {
    Matrix parameters;
    Matrix gradients;
    Vector inner_potential_derivative;
} Layer;


void init_layer(Layer *layer, int x_dim, int y_dim) {
    init_matrix(&(layer->parameters), x_dim + 1, y_dim);
    init_matrix(&(layer->gradients), x_dim + 1, y_dim);
    init_vector(&(layer->inner_potential_derivative), y_dim); 
    for(int y = 0; y < layer->parameters.y_dim; y++) {
        for(int x = 0; x < layer->parameters.x_dim; x++) {    
            set_matrix(layer->parameters, x, y, (float) rand() / RAND_MAX);
        }
    }
}


void delete_layer(Layer layer) {
    delete_matrix(layer.parameters);
    delete_matrix(layer.gradients);
    delete_vector(layer.inner_potential_derivative);
}


typedef struct Output {
    Vector probs;
    FLOAT error;
} Output;



void inplace_softmax(Vector vec) {
    FLOAT sm = 0.0;
    for (int x = 0; x < vec.x_dim; x++) {
        FLOAT val = exp(get_vector(vec, x));
        set_vector(vec, x, val);
        sm += val;
    }
    for (int x = 0; x < vec.x_dim; x++) {
        set_vector(vec, x, get_vector(vec, x) / sm);
    }   
}


FLOAT cross_entropy(Vector vec, Vector target) {
    FLOAT error = 0.0;
    for (int x=0; x < vec.x_dim; x++) {
        error -= log(get_vector(vec, x)) * get_vector(target, x);
    }
    return error;
}


void relu_layer_forward(Matrix params, Vector input, Vector output) {
    for (int y=0; y < params.y_dim; y++) {
        FLOAT sm = get_matrix(params, 0, y);
        for (int x=1; x < params.x_dim; x++) {
            sm += get_matrix(params, x, y) * get_vector(input, x - 1);
        }
        set_vector(output, y, (0.0 < sm) ? sm: 0.0);
    }
}


void relu_layer_backward(Matrix grads, Vector inner_der, Vector input, Vector output) {
    for (int y=0; y < grads.y_dim; y++) {
        if (0.0 < get_vector(output, y)) {
            add_matrix(grads, 0, y, 1.0);
            set_vector(inner_der, y, 1.0);
            for (int x=1; x < grads.x_dim; x++) {
                add_matrix(grads, x, y, get_vector(input, x-1));
            }
        } else {
            set_vector(inner_der, y, 0.0);
        }
    }
}


Vector relu_layer(Layer layer, Vector input, bool grad) {
    Vector new;
    init_vector(&new, layer.parameters.y_dim);
    ASSERT(input.x_dim + 1 == layer.parameters.x_dim,
        "Non-matching matrix sizes %d x %d, %d.\n",
        layer.parameters.x_dim, layer.parameters.y_dim, input.x_dim
    );
    relu_layer_forward(layer.parameters, input, new);
    if (grad) {
        relu_layer_backward(layer.gradients, layer.inner_potential_derivative, input, new);
    }
    delete_vector(input);
    return new;
}


void ces_layer_forward(Matrix params, Vector input, Vector output) {
    for (int y=0; y < params.y_dim; y++) {
        FLOAT sm = get_matrix(params, 0, y);
        for (int x=1; x < params.x_dim; x++) {
            sm += get_matrix(params, x, y) * get_vector(input, x - 1);
        }
        set_vector(output, y, sm);
    }
}


void ces_layer_backward(Matrix grads, Vector inner_der, Vector input, Vector output, Vector target) {
    for (int y=0; y < grads.y_dim; y++) {
        FLOAT val = get_vector(output, y) - get_vector(target, y);
        set_vector(inner_der, y, val);
        add_matrix(grads, 0, y, val);
        for (int x=1; x < grads.x_dim; x++) {
            add_matrix(grads, x, y, val * get_vector(input, x - 1));
        }
    }
}


Output ces_layer(Layer layer, Vector input, Vector *target, bool grad) {
    Vector new;
    init_vector(&new, layer.parameters.y_dim);
    ASSERT(input.x_dim + 1 == layer.parameters.x_dim,
        "Non-matching matrix sizes %d x %d, %d.\n",
        layer.parameters.x_dim, layer.parameters.y_dim, input.x_dim
    );
    ces_layer_forward(layer.parameters, input, new);
    inplace_softmax(new);
    Output out = {.probs=new, .error=0.0};
    if (grad) {
        ASSERT(target != NULL, "Cannot compute gradient, target was not provided.\n");
        out.error = cross_entropy(new, *target);
        ces_layer_backward(layer.gradients, layer.inner_potential_derivative, input, new, *target);
    }

    delete_vector(input);
    return out;
}


#endif