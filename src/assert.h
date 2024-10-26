
#ifndef ASSERT_H
#define ASSERT_H


#define ASSERT(condition, format, ...) \
    do { \
        if (!(condition)) { \
            printf("Assertion failed: " format "\n", ##__VA_ARGS__); \
            exit(1); \
        } \
    } while (0)


#endif