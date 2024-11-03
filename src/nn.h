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



FLOAT sample_normal(FLOAT mean, FLOAT stddev) {
    FLOAT u1 = (FLOAT) rand() / (FLOAT) RAND_MAX;
    FLOAT u2 = (FLOAT) rand() / (FLOAT) RAND_MAX;
    FLOAT pi = (FLOAT) (4 * atan(1.0));
    FLOAT z0 = (FLOAT) (sqrt(-2.f * log(u1)) * cos(2.f * pi * u2));
    return z0 * stddev + mean;
}

void HE_initialization(Layer layer) {
    FLOAT std = 2.f / (FLOAT) layer.parameters.x_dim;
    for(int y = 0; y < layer.parameters.y_dim; y++) {
        for(int x = 0; x < layer.parameters.x_dim; x++) {    
            set_matrix(layer.parameters, x, y, sample_normal(0, std));
        }
    }
}


void glorot_initialization(Layer layer) {
    FLOAT range = (FLOAT) sqrt(6.f / (FLOAT) (layer.parameters.x_dim + layer.parameters.y_dim));
    for(int y = 0; y < layer.parameters.y_dim; y++) {
        for(int x = 0; x < layer.parameters.x_dim; x++) {    
            set_matrix(layer.parameters, x, y, ((FLOAT) rand() / (FLOAT) RAND_MAX) * 2.f * range - range);
        }
    }
}


void init_layer(Layer *layer, int x_dim, int y_dim){
    init_matrix(&(layer->parameters), x_dim + 1, y_dim);
    init_matrix(&(layer->gradients), x_dim + 1, y_dim);
    layer->inner_potential_derivative.data = NULL;
    layer->input.data = NULL;
}


void delete_layer(Layer layer) {
    delete_matrix(layer.parameters);
    delete_matrix(layer.gradients);
}


Vector softmax(Vector vec) {
    Vector new;
    init_vector(&new, vec.x_dim);
    FLOAT sm = 0.f;
    for (int x = 0; x < vec.x_dim; x++) {
        FLOAT val = (FLOAT) exp(get_vector(vec, x));
        // printf("%f %f\n", val, get_vector(vec, x));
        set_vector(new, x, val);
        sm += val;
    }
    for (int x = 0; x < vec.x_dim; x++) {
        set_vector(new, x, get_vector(new, x) / sm);
    }
    return new;
}


FLOAT cross_entropy(Vector vec, Vector target) {
    FLOAT error = 0.f;
    for (int x=0; x < vec.x_dim; x++) {
        error -= (FLOAT) log(get_vector(vec, x)) * get_vector(target, x);
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


void normalize(Vector vec) {
    FLOAT sum = 0.f;
    FLOAT squares = 0.f;
    for (int x = 0; x<vec.x_dim; x++) {
        sum += get_vector(vec, x);
        squares += get_vector(vec, x) * get_vector(vec, x);
    }
    FLOAT mean = sum / (FLOAT) vec.x_dim;
    FLOAT std = (FLOAT) sqrt(squares /  (FLOAT) vec.x_dim - mean * mean);
    for (int x = 0; x<vec.x_dim; x++) {
        FLOAT val = (get_vector(vec, x) - mean) / std;
        set_vector(vec, x, val);
    }
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
    // normalize(new);
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
        ASSERT(grad == false, "ERROR3.\n");
        delete_vector(input);
        return out;
    }
    out.error = cross_entropy(soft, *target);
    if (grad) {
        layer->input = input;
        layer->inner_potential_derivative = ces_der(soft, *target);
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


void init_model(Model *model, int *hidden, int n_hidden, int n_classes, unsigned int seed) {
    model->n_layers = n_hidden;
    MALLOC(model->layers, Layer, (unsigned int) model->n_layers);
    srand(seed);
    for (int i=0;i<n_hidden-1;i++) {
        init_layer(&(model->layers[i]), hidden[i], hidden[i+1]);
        HE_initialization(model->layers[i]);
    }
    init_layer(&(model->layers[n_hidden-1]), hidden[n_hidden-1], n_classes);
    glorot_initialization(model->layers[n_hidden-1]);
}


void delete_model(Model model) {
    for (int i=0;i<model.n_layers;i++) {
        // ASSERT(
        //     model.layers[i].inner_potential_derivative.data == NULL,
        //     "ERROR2.\n"
        // );
        // ASSERT(
        //     model.layers[i].input.data == NULL,
        //     "ERROR1.\n"
        // );
        delete_layer(model.layers[i]);
    }
    FREE(model.layers);
}


void zero_grad(Model model) {
    for (int i=0; i<model.n_layers;i++) {
        Matrix grads = model.layers[i].gradients;
        memset(grads.data, 0.0, (unsigned int) (grads.x_dim * grads.y_dim) * sizeof(FLOAT));
    }
}


void optimize(Model model, float lr, int batch_size) {
    for (int i=0; i<model.n_layers;i++) {
        Matrix grads = model.layers[i].gradients;
        Matrix params = model.layers[i].parameters;
        for (int y=0; y<params.y_dim; y++) {
            for (int x=0; x< params.x_dim; x++) {
                FLOAT val = - lr * get_matrix(grads, x, y) /  (FLOAT) batch_size;
                add_matrix(params, x, y, val);
            }
        }
    }
}


void backprop(Model model) {
    Layer last = model.layers[model.n_layers-1];
    Vector temp1;
    Vector temp2;

    for (int y=0; y<last.parameters.y_dim;y++) {
        FLOAT val = get_vector(last.inner_potential_derivative, y);
        add_matrix(last.gradients, 0, y, val);
        for (int x=1;x<last.parameters.x_dim;x++) {
            add_matrix(last.gradients, x, y, val * get_vector(last.input, x-1));
        }
    }
    
    init_vector(&temp2, last.parameters.y_dim);
    for (int y=0; y<last.parameters.y_dim; y++) {
        set_vector(temp2, y, 1.0);
    }

    for (int i=model.n_layers-2; 0<=i;i--) {
        
        Layer layer1 = model.layers[i];
        Layer layer2 = model.layers[i+1];

        init_vector(&temp1, layer2.parameters.x_dim);
        for (int y=0; y<layer2.parameters.y_dim; y++) {
            for (int x=1; x<layer2.parameters.x_dim; x++) {
                FLOAT val = get_vector(temp2, y);
                val *= get_vector(layer2.inner_potential_derivative, y);
                add_vector(temp1, x-1, val * get_matrix(layer2.parameters, x, y));
            }
        }

        for (int y=0; y<layer1.parameters.y_dim;y++) {
            FLOAT val = get_vector(layer1.inner_potential_derivative, y);
            val *= get_vector(temp1, y);
            add_matrix(layer1.gradients, 0, y, val);
            for (int x=1;x<layer1.parameters.x_dim;x++) {
                add_matrix(layer1.gradients, x, y, val * get_vector(layer1.input, x-1));
            }
        }
        
        delete_vector(temp2);
        delete_vector(layer2.input);
        delete_vector(layer2.inner_potential_derivative);
        temp2 = temp1;
    }
    delete_vector(temp1);
    delete_vector(model.layers[0].input);
    delete_vector(model.layers[0].inner_potential_derivative);
}


Output forward(Model model, Vector input, Vector *target, bool grad) {
    for (int i=0; i<model.n_layers-1;i++) {
        input = relu_layer_forward(&model.layers[i], input, grad);
    }
    return ces_layer_forward(&model.layers[model.n_layers-1], target, input, grad);
}

#endif