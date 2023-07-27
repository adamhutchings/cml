#include "iris-reader.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * CODE: Setosa is 0.1, Versicolor is 0.5, and Virginica is 0.9
 */
static float get_type(char * buf) {
    if (strcmp(buf, "Iris-setosa") == 0) {
        return 0.1f;
    } else if (strcmp(buf, "Iris-versicolor") == 0) {
        return 0.5f;
    }
    if (strcmp(buf, "Iris-virginica") == 0) {
        return 0.9f;
    }
    /* We better never reach here. */
    assert(0);
}

int cmlreadiris(const char * filepath, struct cmlvector ** ins, struct cmlvector ** outs, int * size) {

    static int INPUT_SIZE = 4;

    FILE * fp;
    char namebuf[64];

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

    *ins = calloc(l, sizeof (struct cmlvector));
    *outs = calloc(l, sizeof (struct cmlvector));
    assert(*ins);
    assert(*outs);

    /* Start at the beginning of the file and read every line. */
    rewind(fp);

    for (int i = 0; i < l; ++i) {

        /* Read four floating-point numbers and then a comma. */
        cmlvinit(&((*ins)[i]), INPUT_SIZE);
        for (int j = 0; j < INPUT_SIZE; ++j) {
            fscanf(fp, "%f", &((*ins)[i].entries[j]));
            assert(getc(fp) == ',');
        }

        /* Read the string and determine what it is. */
        cmlvinit(&((*outs)[i]), 1);
        fscanf(fp, "%s", namebuf);
        (*outs)[i].entries[0] = get_type(namebuf);

        assert(getc(fp) == '\n');

    }

    fclose(fp);

    return 0;

}

int cmlfreedata(struct cmlvector * ins, struct cmlvector * outs, int no) {
    assert(ins);
    assert(outs);
    for (int i = 0; i < no; ++i) {
        cmlvfree(&(ins[i]));
        cmlvfree(&(outs[i]));
    }
    free(ins);
    free(outs);
    return 0;
}
