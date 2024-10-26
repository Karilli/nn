#include "csv.h"
#include "array.h"
#include "nn.h"
#include "macro_utils.h"

#include <stdlib.h>
#include "math.h"


int main(void) {
    Matrix inputs;
    read_matrix(&inputs, "data/fashion_mnist_train_vectors.csv", 28*28);

    Parameters layer1;
    Parameters layer2;
    
    return 0;
}
