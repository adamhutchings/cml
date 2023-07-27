#include "iris-reader.h"

#include <assert.h>
#include <stdio.h>

int cmlreadiris(const char * filepath, struct cmlvector ** ins, struct cmlvector ** outs, int * size) {

    FILE * fp;

    /* First, do a scan-through of the file to get the length. */
    int l = 0;

    fp = fopen(filepath, "r");
    assert(fp);

    int c;
    while ((c = getc(fp)) != EOF) {
        if (c == '\n')
            ++l;
    }
    *size = l;

    return 0;

}
