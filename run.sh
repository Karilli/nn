#!/bin/bash
## change this file to your needs

echo "#################"
echo "  CREATING ENV   "
echo "#################"

# echo "Adding some modules"

# module add gcc-10.2


echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
# g++ -Wall -std=c++17 -O3 src/main.cpp src/file2.cpp -o network

echo "test1"
gcc -Wall -Werror -O3 -o test1.o test/test1.c -lm -Isrc
echo "main"
gcc -Wall -Werror -O3 -o main.o src/main.c -lm

echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
# nice -n 19 ./network

echo "test1"
./test1.o
echo "main"
./main.o

echo "#################"
echo "     CLEANUP     "
echo "#################"

echo "removing object files"
rm *.o
