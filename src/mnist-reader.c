#include "mnist-reader.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cmlreadmnist(const char * filepath, struct cmlvector ** ins, struct cmlvector ** outs, int * size) {

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

    *ins = calloc(l, sizeof (struct cmlvector));
    *outs = calloc(l, sizeof (struct cmlvector));
    assert(*ins);
    assert(*outs);

    /* Start at the beginning of the file and read every line. */
    rewind(fp);

    for (int i = 0; i < l; ++i) {

        int o;

        /* Read the first number. */
        cmlvinit(&((*outs)[i]), MNIST_OUTPUT_SIZE);
        fscanf(fp, "%d", &o);
        /* One-hot system -- outputs are an array of 10, of which 1 is on. */
        (*outs)[i].entries[o] = 1;

        /* Read the inputs. */
        cmlvinit(&((*ins)[i]), MNIST_INPUT_SIZE);
        for (int j = 0; j < MNIST_INPUT_SIZE; ++j) {
            assert(getc(fp) == ',');
            fscanf(fp, "%f", &((*ins)[i].entries[j]));
            (*ins)[i].entries[j] /= 256; /* Normalizing data. */
        }

        assert(getc(fp) == '\n');

    }

    fclose(fp);

    return 0;

}

int cmlfreemnist(struct cmlvector * ins, struct cmlvector * outs, int no) {
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
