#ifndef CSV_H
#define CSV_H

#include "matrix.h"
#include "precision.h"
#include "macro_utils.h"


#include <string.h>
#include <stdio.h>
#include <stdlib.h>


int num_of_lines(FILE *file, int cols) {
    // Assuming integers in range 0-255 (three digit numbers)
    char *line;
    MALLOC(line, char, cols*4);
    int line_count = 0;
    while (fgets(line, cols*4, file)) {
        line_count++;
    }
    fseek(file, 0, SEEK_SET);
    FREE(line);
    return line_count;
}


int read_matrix(Matrix *mat, char* filepath, int cols) {
    FILE *file = fopen(filepath, "r");
    ASSERT(file != NULL, "Could not open file %s.\n", filepath);

    init_matrix(mat, cols, num_of_lines(file, cols));
    
    int x = 0;
    int y = 0;
    char c;
    int curr_value = 0;
    while ((c = fgetc(file)) != EOF) {
        if (c == ',') {
            set_matrix(*mat, x, y, (FLOAT) curr_value);
            curr_value = 0;
            x++;
        } else if (c == '\n') {
            ASSERT(x == cols - 1, "Expected %d number of columns, found %d.\n", cols, x+1);
            set_matrix(*mat, x, y, (FLOAT) curr_value);
            curr_value = 0;
            x = 0;
            y++;
        } else {
            curr_value = 10 * curr_value + (c - '0');
        }
    }
    ASSERT(x == 0, "EOF encountered before '\\n', could not finnish reading a row.\n");
    ASSERT(curr_value == 0, "Encounterd EOF before '\\n' or ','.\n");
    
    fclose(file);
    return 0;
}

#endif