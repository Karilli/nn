#ifndef NN_H
#define NN_H


#include "csv.h"
#include "array.h"
#include "assert.h"

#include <stdlib.h>
#include <time.h>
#include "math.h"
#include "stdbool.h"


typedef struct Layer {
    Matrix parameters;
    Matrix gradients;
    Vector inner_potential_derivative;
    Vector input;
} Layer;


typedef struct Output {
    Vector probs;
    FLOAT error;
} Output;



typedef struct Model {
    Layer *layers;
    int n_layers;
} Model;


void init_layer(Layer *layer, int x_dim, int y_dim){
    init_matrix(&(layer->parameters), x_dim + 1, y_dim);
    init_matrix(&(layer->gradients), x_dim + 1, y_dim);
    layer->inner_potential_derivative.data = NULL;
    layer->input.data = NULL;
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
    return new;
}


FLOAT cross_entropy(Vector vec, Vector target) {
    FLOAT error = 0;
    for (int x=0; x < vec.x_dim; x++) {
        error -= log(get_vector(vec, x)) * get_vector(target, x);
    }
    return error;
}


Vector relu(Vector vec) {
    Vector new;
    init_vector(&new, vec.x_dim);

    for (int x = 0; x < vec.x_dim; x++) {
        FLOAT val = get_vector(vec, x);
        set_vector(new, x, (0 <= val) ? val : 0);
    } 

    return new;
}


Vector matmul(Matrix mat, Vector vec) {
    Vector new;
    init_vector(&new, mat.y_dim);
    ASSERT(vec.x_dim + 1 == mat.x_dim,
        "Non-matching matrix sizes %d x %d, %d",
        mat.x_dim, mat.y_dim, vec.x_dim
    );
    for (int y=0; y < mat.y_dim; y++) {
        FLOAT sm = get_matrix(mat, 0, y);
        for (int x=1; x < mat.x_dim; x++) {
            FLOAT val = get_matrix(mat, x, y) * get_vector(vec, x - 1);
            sm += val;
        }
        set_vector(new, y, sm);
    }
    return new;
}


Vector relu_der(Vector inner) {
    Vector new;
    init_vector(&new, inner.x_dim);
    for (int x=0; x<inner.x_dim;x++) {
        set_vector(new, x, 0.0 <= get_vector(inner, x) ? 1.0 : 0.0);
    }
    return new;
}


Vector ces_der(Vector soft, Vector target) {
    Vector new;
    init_vector(&new, soft.x_dim);
    for (int x=0; x<soft.x_dim;x++) {
        FLOAT val = get_vector(soft, x) - get_vector(target, x);
        set_vector(new, x, val);
    }
    return new;
}


Output ces_layer_forward(Layer *layer, Vector *target, Vector input, bool grad) {
    Vector inner = matmul(layer->parameters, input);
    Vector soft = softmax(inner);
    Output out = {.probs = soft, .error=0.0};
    delete_vector(inner);
    if (target == NULL) {
        ASSERT(grad == false, "ERROR.\n");
        delete_vector(input);
        return out;
    }
    out.error = cross_entropy(soft, *target);
    if (grad) {
        printf("hello world\n");
        printf("%d\n", input.x_dim);
        layer->input = input;
        printf("%d\n", layer->input.x_dim);
        layer->inner_potential_derivative = ces_der(soft, *target);
        printf("%d\n", layer->inner_potential_derivative.x_dim);
    } else {
        delete_vector(input);
    }
    return out;
}


Vector relu_layer_forward(Layer *layer, Vector input, bool grad) {
    Vector inner = matmul(layer->parameters, input);
    Vector activation = relu(inner);
    if (grad) {
        layer->input = input;
        layer->inner_potential_derivative = relu_der(inner);
    } else {
        delete_vector(input);
    }
    delete_vector(inner);
    return activation;
}


void init_model(Model *model, int *hidden, int n_hidden, int n_classes) {
    model->n_layers = n_hidden;
    MALLOC(model->layers, Layer, model->n_layers);
    for (int i=0;i<n_hidden-1;i++) {
        init_layer(&(model->layers[i]), hidden[i], hidden[i+1]);
    }
    init_layer(&(model->layers[n_hidden-1]), hidden[n_hidden-1], n_classes);
}


void delete_model(Model model) {
    for (int i=0;i<model.n_layers;i++) {
        ASSERT(
            model.layers[i].inner_potential_derivative.data == NULL,
            "ERROR.\n"
        );
        ASSERT(
            model.layers[i].input.data == NULL,
            "ERROR.\n"
        );
        delete_layer(model.layers[i]);
    }
}


// void zero_grad(Model mode) {

// }


void backprop(Model model) {
    Layer last = model.layers[model.n_layers-1];

    print_vector(last.inner_potential_derivative);
    print_vector(last.input);

    for (int y=0; y<last.parameters.y_dim;y++) {
        FLOAT val = get_vector(last.inner_potential_derivative, y);
        
        set_matrix(last.gradients, 0, y, val);
        for (int x=1;x<last.parameters.x_dim;x++) {
            set_matrix(last.gradients, x, y, val * get_vector(last.input, x-1));
        }
    }
    for (int i=model.n_layers-1; 0<=i;i--) {

    }
}


#endif