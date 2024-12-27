
#ifndef CSV_H
#define CSV_H

#include "array.h"
#include "macro_utils.h"


#include <string.h>
#include <stdio.h>
#include <stdlib.h>


void get_csv_dimensions(FILE *file, int *rows, int *cols) {
    int c;
    int comma_count = 0;
    int enter_count = 0;
    while ((c = fgetc(file)) != EOF) {
        comma_count += c == ',';
        enter_count += c == '\n';
    }

    *rows = enter_count;
    *cols = (comma_count + enter_count) / enter_count;
    
    fseek(file, 0, SEEK_SET);
}


int read_matrix(Matrix *mat, char* filepath) {
    FILE *file = fopen(filepath, "r");
    ASSERT(file != NULL, "Could not open file %s.", filepath);

    int rows;
    int cols;
    get_csv_dimensions(file, &rows, &cols);
    printf("Reading csv '%s' with dimensions: %d x %d\n", filepath, cols, rows);
    init_matrix(mat, cols, rows);
    
    int x = 0;
    int y = 0;
    char c;
    int curr_value = 0;
    while ((c = (char) fgetc(file)) != EOF) {
        if (c == ',') {
            set_matrix(*mat, x, y, (FLOAT) curr_value);
            curr_value = 0;
            x++;
        } else if (c == '\n') {
            ASSERT(x == cols - 1, "Expected %d number of columns, found %d.", cols, x+1);
            set_matrix(*mat, x, y, (FLOAT) curr_value);
            curr_value = 0;
            x = 0;
            y++;
        } else {
            curr_value = 10 * curr_value + (c - '0');
        }
    }
    ASSERT(x == 0, "EOF encountered before '\\n', could not finnish reading a row.");
    ASSERT(curr_value == 0, "Encounterd EOF before '\\n' or ','.");
    
    fclose(file);
    return 0;
}


void write_csv(Matrix mat, char* filepath) {
    printf("Writing csv '%s' with dimensions: %d x %d\n", filepath, mat.x_dim, mat.y_dim);
    FILE *file = fopen(filepath, "w");
    for (int y=0; y<mat.y_dim; y++) {
        fprintf(file,"%d\n", (int) get_matrix(mat, 0, y));
    }
    fclose(file);
}
#endif