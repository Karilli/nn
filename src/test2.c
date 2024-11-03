#include "nn.h"
#include "array.h"
#include "macro_utils.h"
#include "math.h"


int main(void) {
    Vector input;
    init_vector(&input, 2);
    input.data[0] = 0.5;
    input.data[1] = 0.75;

    Vector target;
    init_vector(&target, 2);
    target.data[0] = 0;
    target.data[1] = 1;

    Model model;
    int hidden[2] = {2, 3};
    init_model(&model, hidden, 2, 2, 0);
    Layer *layer1 = &model.layers[0];


    layer1->parameters.data[0] = 0.0;
    layer1->parameters.data[1] = 0.5;
    layer1->parameters.data[2] = 0.3;

    layer1->parameters.data[3] = 0.0;
    layer1->parameters.data[4] = 0.2;
    layer1->parameters.data[5] = 0.1;

    layer1->parameters.data[6] = 0.0;
    layer1->parameters.data[7] = 0.7;
    layer1->parameters.data[8] = 0.9;


    Layer *layer2 = &model.layers[1];
    layer2->parameters.data[0] = 0.0;
    layer2->parameters.data[1] = 0.5;
    layer2->parameters.data[2] = 0.3;
    layer2->parameters.data[3] = 0.2;

    layer2->parameters.data[4] = 0.0;
    layer2->parameters.data[5] = 0.1;
    layer2->parameters.data[6] = 0.7;
    layer2->parameters.data[7] = 0.9;


    input = relu_layer_forward(layer1, input, 1);
    FLOAT data1[3] = {0.475000, 0.175000, 1.025000};
    assert_vector_equals(input, data1, 3, 1e-6);

    Output out = ces_layer_forward(layer2, &target, input, 1);
    Vector probs = out.probs;
    FLOAT error = out.error;

    backprop(model);

    FLOAT data2[2] = {0.354916, 0.645084};
    assert_vector_equals(probs, data2, 2, 1e-6);

    FLOAT data4[8] = {
        0.354916, 0.168585, 0.062110, 0.363789,
        -0.354916, -0.168585, -0.062110, -0.363789,
    };
    assert_matrix_equals(layer2->gradients, data4, 4, 2, 1e-6);
    FLOAT data3[9] = {
        0.141966, 0.070983, 0.106475,
        -0.141966, -0.070983, -0.106475,
        -0.248441, -0.124221, -0.186331,
    };
    assert_matrix_equals(layer1->gradients, data3, 3, 3, 1e-6);


    ASSERT(
        fabs(error - 0.438375f) < 0.0001f,
        "Expected cross entropy error %f, but got %f.\n",
        0.438375, error
    );

    delete_vector(probs);
    delete_vector(target);
    delete_model(model);
    ASSERT(
        ALLOC_COUNTER == 0,
        "Expected ALLOC_COUNTER to be 0, but got %d.\n",
        ALLOC_COUNTER
    );
    return 0;
}