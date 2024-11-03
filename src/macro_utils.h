
#ifndef ASSERT_H
#define ASSERT_H


int ALLOC_COUNTER = 0;


#ifndef PROD 
    #define ASSERT(condition, format, ...) \
        do { \
            if (!(condition)) { \
                printf(format, ##__VA_ARGS__); \
                exit(1); \
            } \
        } while (0)
#else
    #define ASSERT(condition, format, ...)
#endif


#define MALLOC(ptr, type, size) \
    do { \
        (ptr) = (type *)malloc((size) * sizeof(type)); \
        if ((ptr) == NULL) { \
            printf("Error: Memory allocation failed.\n"); \
            exit(1); \
        } \
        ALLOC_COUNTER++; \
    } while(0)


#define FREE(ptr) \
    do { \
        if ((ptr) == NULL) { \
            printf("Error: Attempt to free a NULL pointer.\n"); \
            exit(1); \
        } \
        free(ptr); \
        (ptr) = NULL; \
        ALLOC_COUNTER--; \
    } while(0)


#endif