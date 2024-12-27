#!/bin/bash


OPTIONS="-Wall -Werror -Wconversion -Wextra"
gcc -o main.o -Ofast -D PROD $OPTIONS src/main.c -lm
nice -n 19 ./main.o
echo "train accuracy:"
python3 evaluator/evaluate.py ./train_predictions.csv data/fashion_mnist_train_labels.csv
echo "test accuracy:"
python3 evaluator/evaluate.py ./test_predictions.csv data/fashion_mnist_test_labels.csv
