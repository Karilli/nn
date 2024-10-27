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
    target.data[0] = 0.0;
    target.data[1] = 1.0;

    Layer layer1;
    init_layer(&layer1, 2, 3);
    layer1.parameters.data[0] = 0.0;
    layer1.parameters.data[1] = 0.5;
    layer1.parameters.data[2] = 0.3;

    layer1.parameters.data[3] = 0.0;
    layer1.parameters.data[4] = 0.2;
    layer1.parameters.data[5] = 0.1;

    layer1.parameters.data[6] = 0.0;
    layer1.parameters.data[7] = 0.7;
    layer1.parameters.data[8] = 0.9;

    Layer layer2;
    init_layer(&layer2, 3, 2);
    layer2.parameters.data[0] = 0.0;
    layer2.parameters.data[1] = 0.5;
    layer2.parameters.data[2] = 0.3;
    layer2.parameters.data[3] = 0.2;

    layer2.parameters.data[4] = 0.0;
    layer2.parameters.data[5] = 0.1;
    layer2.parameters.data[6] = 0.7;
    layer2.parameters.data[7] = 0.9;

    input = propagate_through_layer(layer1.parameters, input);
    input = relu(input);
    input = propagate_through_layer(layer2.parameters, input);
    input = relu(input);
    input = softmax(input);
    FLOAT error = cross_entropy(input, target);

    ASSERT(
        fabs(error - 0.438375) < 1e-6,
        "Expected cross entropy error %f, but got %f.\n",
        0.438375, error
    );

    delete_vector(target);
    delete_layer(layer1);
    delete_layer(layer2);
    ASSERT(
        ALLOC_COUNTER == 0,
        "Expected ALLOC_COUNTER to be 0, but got %d.",
        ALLOC_COUNTER
    );
    return 0;
}