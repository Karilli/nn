#include "csv.h"
#include "array.h"
#include "nn.h"
#include "macro_utils.h"

#include <stdlib.h>
#include "math.h"


#define LEARNING_RATE 0.001f
#define BATCH_SIZE 1024
#define EPOCHS 100
#define SEED 0
#define INPUT_SIZE 28*28
#define MEAN 0.286f
#define STD 0.35f


void read_input(Matrix inputs, int y, Vector *input) {
    init_vector(input, 28*28);
    for (int x=0; x<inputs.x_dim; x++) {
        set_vector(*input, x, (get_matrix(inputs, x, y) / 255.f - MEAN) / STD);
    }
}


void read_target(Matrix targets, int y, Vector *target) {
    init_vector(target, 10);
    set_vector(*target, (int) get_matrix(targets, 0, y), 1.0);
}


int argmax(Vector probs) {
    FLOAT max = 0.0;
    int res = 0;
    for (int x=0; x<probs.x_dim; x++) {
        FLOAT val = get_vector(probs, x);
        if (max < val) {
            max = val;
            res = x;
        }
    }
    return res;
}


void train_loop(Model model, Matrix inputs, Matrix targets, unsigned int seed) {
    srand(seed);
    
    for (int epoch=0; epoch < EPOCHS; epoch++) {
        zero_grad(model);
        FLOAT error = 0.0;
        for (int i=0; i< BATCH_SIZE; i++) {
            Vector input;
            Vector target;
            int y = rand() % inputs.y_dim;
            read_input(inputs, y, &input);
            read_target(targets, y, &target);

            Output out = forward(model, input, &target, 1);
            backprop(model);
            error += out.error;
            delete_vector(out.probs);
            delete_vector(target);
        }
        optimize(model, LEARNING_RATE, BATCH_SIZE);
        printf("epoch %d: %f\n", epoch, error / BATCH_SIZE);
        error = 0.0;
    }
}


void test_loop(Model model, Matrix inputs, Matrix targets) {
    for (int y=0; y<inputs.y_dim; y++) {
        Vector input;
        read_input(inputs, y, &input);
        Output out = forward(model, input, NULL, 0);
        set_matrix(targets, 0, y, (FLOAT) argmax(out.probs));
        delete_vector(out.probs);
    }
}


int main(void) {
    Matrix inputs;
    Matrix targets;
    Model model;

    int hidden[] = {INPUT_SIZE, 10};
    int size = sizeof(hidden) / sizeof(hidden[0]);
    init_model(&model, hidden, size, 10, SEED);
    
    read_matrix(&inputs, "data/fashion_mnist_train_vectors.csv", INPUT_SIZE);
    read_matrix(&targets, "data/fashion_mnist_train_labels.csv", 1);
    train_loop(model, inputs, targets, SEED);
    delete_matrix(inputs);
    delete_matrix(targets);

    read_matrix(&inputs, "data/fashion_mnist_test_vectors.csv", INPUT_SIZE);
    init_matrix(&targets, 1, inputs.y_dim);
    test_loop(model, inputs, targets);
    write_csv(targets, "result.csv");
    delete_matrix(targets);
    delete_matrix(inputs);

    delete_model(model);
    ASSERT(
        ALLOC_COUNTER == 0,
        "Expected ALLOC_COUNTER to be 0, but got %d.\n",
        ALLOC_COUNTER
    );
    return 0;
}
