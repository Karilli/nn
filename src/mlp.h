#ifndef MLP_H
#define MLP_H


#include "csv.h"
#include "vector.h"
#include "matrix.h"
#include "macro_utils.h"
#include "precision.h"
#include "layer.h"

#include <stdlib.h>
#include <time.h>
#include "math.h"
#include "stdbool.h"


typedef struct MLP {
    Layer *layers;
    int n_layers;
} MLP;


void init_model(MLP *mlp, int n_hidden, int n_classes, int *hidden_sizes) {
    mlp->n_layers = n_hidden + 1;
    MALLOC(mlp->layers, Layer, mlp->n_layers);
    for (int i=0; i<mlp->n_layers-1;i++) {
        init_layer(&(mlp->layers[i]), hidden_sizes[i], hidden_sizes[i+1]);
    }
    init_layer(&(mlp->layers[mlp->n_layers-1]), hidden_sizes[mlp->n_layers-1], n_classes);
}


void delete_model(MLP mlp) {
    for (int i=0; i<mlp.n_layers;i++) {
        delete_layer(mlp.layers[i]);
    }
}


void _backprop(Layer layer1, Layer layer2) {
    Vector temp;
    init_vector(&temp, layer2.parameters.x_dim - 1);
    printf("hello\n");
    for (int y=0; y < layer2.parameters.y_dim; y++) {
        for (int x=1; x < layer2.parameters.x_dim; x++) {
            FLOAT val = get_matrix(layer2.parameters, x, y) * get_vector(layer2.inner_potential_derivative, y);
            add_vector(temp, x-1, val);
        }
    }
    printf("hello\n");
    for (int y=0; y < layer1.gradients.y_dim; y++) {
        FLOAT val = get_vector(layer1.inner_potential_derivative, 0) * get_vector(temp, y);
        set_matrix(layer1.gradients, 0, y, val);
        for (int x=1; x < layer1.gradients.x_dim; x++) {
            val = get_vector(layer1.inner_potential_derivative, x) * get_vector(layer1.input, x-1) * get_vector(temp, y);
            set_matrix(layer1.gradients, x, y, val);
        }
    }
    delete_vector(temp);
}


Output forward(MLP mlp, Vector input, Vector *target, bool grad) {
    for (int i=0; i<mlp.n_layers - 1; i++) {
        input = relu_layer(mlp.layers[i], input, grad);
    }
    return ces_layer(mlp.layers[mlp.n_layers-1], input, target, grad);
}


void backward(MLP mlp) {
    printf("hello");
    for (int i=mlp.n_layers - 2; 0 <= i; i--) {
        _backprop(mlp.layers[i], mlp.layers[i+1]);
    }
    for (int i=0; i<mlp.n_layers; i++) {
        delete_vector(mlp.layers[i].inner_potential_derivative);
        delete_vector(mlp.layers[i].input);
    }
}


void zero_grad(MLP mlp) {
    for (int i=0; i<mlp.n_layers; i++) {
        memset(mlp.layers[i].gradients.data, 0.0, mlp.layers[i].gradients.x_dim * mlp.layers[i].gradients.y_dim * sizeof(FLOAT));
    }
}
#endif
