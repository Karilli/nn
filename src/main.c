#include "csv.h"
#include "array.h"
#include "nn.h"
#include "macro_utils.h"

#include <stdlib.h>
#include "stdint.h"
#include "math.h"
#include <time.h>


#define BATCH_SIZE 16
#define EPOCHS 120
#define SEED 0
#define NORMALIZE_INPUT 1
#define LEARNING_RATE_1 0.000255f
#define LEARNING_RATE_2 0.000001f
#define TRAIN_TEST_SPLIT 0.9f
#define OPTIMIZER optimize_adam
#define BETA1 0.9f
#define BETA2 0.99f


void read_input(Matrix inputs, int y, Vector *input, FLOAT mean, FLOAT std) {
    init_vector(input, inputs.x_dim);

    for (int x=0; x<inputs.x_dim; x++) {
        if (NORMALIZE_INPUT) {
            set_vector(*input, x, (get_matrix(inputs, x, y) / 255.f - mean) / std);
        } else {
            set_vector(*input, x, get_matrix(inputs, x, y) / 255.f);
        }
    }
}


void read_target(Matrix targets, int y, Vector *target, int n_classes) {
    init_vector(target, n_classes);
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


void train_loop(Model model, Matrix inputs, Vector train_idx, Vector test_idx, Matrix targets, FLOAT mean, FLOAT std, int n_classes) {
    int batches_in_one_epoch = train_idx.x_dim/BATCH_SIZE;

    printf("Started training:\n");
    for (int epoch=0; epoch < EPOCHS; epoch++) {
        clock_t start = clock();

        FLOAT train_error = 0.0f;
        int train_acc = 0;
        FLOAT lr = scheduler(epoch, EPOCHS, LEARNING_RATE_1, LEARNING_RATE_2);
        for (int j=0; j<batches_in_one_epoch; j++) {
            zero_grad(model);
            FLOAT error = 0.0;
            for (int i=0; i< BATCH_SIZE; i++) {
                Vector input;
                Vector target;
                int idx = rand() % train_idx.x_dim;
                int y = (int) get_vector(train_idx, idx);
                
                read_input(inputs, y, &input, mean, std);
                read_target(targets, y, &target, n_classes);

                Output out = forward(model, input, &target, 1);
                backprop(model, BATCH_SIZE);
                error += out.error;
                train_acc += argmax(out.probs) == get_matrix(targets, 0, y);
                delete_vector(out.probs);
            }
            
            OPTIMIZER(model, lr, BETA1, BETA2, BATCH_SIZE);
            train_error += error / BATCH_SIZE;
            error = 0.0;
        }
        FLOAT test_error = 0.0f;
        int test_acc = 0;
        for (int i=0; i < test_idx.x_dim; i++) {
            Vector input;
            Vector target;
            int y = (int) get_vector(test_idx, i);
            read_input(inputs, y, &input, mean, std);
            read_target(targets, y, &target, n_classes);
            Output out = forward(model, input, &target, 0);
            test_error += out.error;
            test_acc += argmax(out.probs) == get_matrix(targets, 0, y);
            delete_vector(out.probs);
        }

        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf(
            "epoch %d: train-error %.3f, test-error %.3f, train-acc %.3f, test-acc %.3f, lr %f, %.3f sec.\n", 
            epoch, 
            train_error / (FLOAT) batches_in_one_epoch, 
            test_error / (FLOAT) test_idx.x_dim, 
            (FLOAT) train_acc / (FLOAT) train_idx.x_dim, 
            (FLOAT) test_acc / (FLOAT) test_idx.x_dim,
            lr, 
            time_spent
        );
    }
}


void predict(Model model, Matrix inputs, Matrix outputs, FLOAT mean, FLOAT std) {
    for (int y=0; y<inputs.y_dim; y++) {
        Vector input;
        read_input(inputs, y, &input, mean, std);
        Output out = forward(model, input, NULL, 0);
        set_matrix(outputs, 0, y, (FLOAT) argmax(out.probs));
        delete_vector(out.probs);
    }
}


void compute_stats(Matrix mat, FLOAT *mean, FLOAT *std) {
    int64_t sum = 0.0f;
    int64_t sum_squares = 0.0f;
    for (int y=0; y < mat.y_dim; y++) {
        for (int x=0; x < mat.x_dim; x++) {
            int64_t val = (int64_t) get_matrix(mat, x, y);
            sum += val;
            sum_squares += val * val;
        }
    }
    *mean = (FLOAT) sum / (FLOAT) (mat.x_dim * mat.y_dim) / 255.f;
    *std = (FLOAT) sqrt( (((FLOAT) sum_squares / (FLOAT) (mat.x_dim * mat.y_dim)) - (*mean) * (*mean) * 255.f * 255.f)) / 255.f;
}


void train_test_split(Matrix inputs, Vector *train_idx, Vector *test_idx, FLOAT ratio) {
    int train = 0;
    int test = 0;
    init_vector(train_idx, (int) (ratio * (FLOAT) inputs.y_dim));
    init_vector(test_idx, inputs.y_dim - train_idx->x_dim);
    ASSERT(train_idx->x_dim + test_idx->x_dim == inputs.y_dim, "error 1\n");
    for (int x = 0; x < inputs.y_dim; x++) {
        ASSERT(train <= train_idx->x_dim, "error 2\n");
        ASSERT(test <= test_idx->x_dim, "error 3\n");
        if (((rand() % RAND_MAX < ratio) && (train < train_idx->x_dim)) || !(test < test_idx->x_dim)) {
            set_vector(*train_idx, train, (FLOAT) x);
            train ++;
        } else if (test < test_idx->x_dim) {
            set_vector(*test_idx, test, (FLOAT) x);
            test ++;
        }
    }
    ASSERT(train == train_idx->x_dim, "aa");
    ASSERT(test == test_idx->x_dim, "aa");
}


int main(void) {
#ifdef PROD
    printf("ASSERTs are disabled.\n");
#else
    printf("ASSERTs are enabled.\n");
#endif

    Matrix inputs;
    Matrix targets;
    Matrix ouputs;
    Model model;

    srand(SEED);

    read_matrix(&inputs, "data/fashion_mnist_train_vectors.csv");
    read_matrix(&targets, "data/fashion_mnist_train_labels.csv");
    int n_classes = (int) matrix_max(targets) + 1;  // Assuming classes 0, 1, 2, .., n_classes - 1
    printf("Number of classes: %d\n", n_classes);

    FLOAT mean;
    FLOAT std;
    compute_stats(inputs, &mean, &std);
    printf("Computed mean: %f, std: %f\n", mean, std);

    Vector train_idx;
    Vector test_idx;
    int hidden[] = {inputs.x_dim, 128, 32, 10};
    int size = sizeof(hidden) / sizeof(hidden[0]);
    init_model(&model, hidden, size, n_classes, SEED);

    train_test_split(inputs, &train_idx, &test_idx, TRAIN_TEST_SPLIT);
    train_loop(model, inputs, train_idx, test_idx, targets, mean, std, n_classes);
    delete_matrix(targets);
    delete_vector(train_idx);
    delete_vector(test_idx);

    init_matrix(&ouputs, 1, inputs.y_dim);
    predict(model, inputs, ouputs, mean, std);
    write_csv(ouputs, "train_predictions.csv");
    delete_matrix(ouputs);
    delete_matrix(inputs);

    read_matrix(&inputs, "data/fashion_mnist_test_vectors.csv");
    init_matrix(&ouputs, 1, inputs.y_dim);
    predict(model, inputs, ouputs, mean, std);
    write_csv(ouputs, "test_predictions.csv");
    delete_matrix(inputs);
    delete_matrix(ouputs);

    delete_model(model);

    ASSERT(
        ALLOC_COUNTER == 0,
        "Expected ALLOC_COUNTER to be 0, but got %d.\n",
        ALLOC_COUNTER
    );
}
