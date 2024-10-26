
#ifndef CSV_H
#define CSV_H

#include "array.h"
#include "assert.h"


#include <string.h>
#include <stdio.h>
#include <stdlib.h>


int num_of_lines(FILE *file, int cols) {
    char *line=malloc(cols*4*sizeof(line));
    int line_count = 0;
    while (fgets(line, cols*4, file)) {
        line_count++;
    }
    fseek(file, 0, SEEK_SET);
    free(line);
    return line_count;
}


int read_matrix(Matrix *mat, char* filepath, int cols){
    FILE *file = fopen(filepath, "r");
    ASSERT(file != NULL, "Could not open file %s.", filepath);

    init_matrix(mat, cols, num_of_lines(file, cols));
    
    int x = 0;
    int y = 0;
    char c;
    int curr_value = 0;
    while ((c = fgetc(file)) != EOF) {
        if (c == ',') {
            set_matrix(mat, x, y, (FLOAT) curr_value);
            curr_value = 0;
            x++;
        } else if (c == '\n') {
            ASSERT(x == cols - 1, "Expected %d number of columns, found %d.", cols, x+1);
            set_matrix(mat, x, y, (FLOAT) curr_value);
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

#endif