#include "csv.h"
#include "mlp.h"

#include <stdlib.h>
#include "math.h"


int main(void) {
    srand(0);

    Matrix inputs;
    read_matrix(&inputs, "data/fashion_mnist_train_vectors.csv", 28*28);

    // Layer layer1;
    // Layer layer2;

    delete_matrix(inputs);
    ASSERT(
        ALLOC_COUNTER == 0,
        "Expected ALLOC_COUNTER to be 0, but got %d.\n",
        ALLOC_COUNTER
    );
    return 0;
}
