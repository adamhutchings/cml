#include "mvmath.h"

#include <assert.h>
#include <stdlib.h>

/* Nothing but initialization of the memory. */
int cmlvinit(struct cmlvector * v, int size) {
    assert(v);
    v->len = size;
    v->entries = calloc(v->len, sizeof(float));
    assert(v->entries);
    return 0;
}

int cmlvfree(struct cmlvector * v) {
    assert(v);
    free(v->entries);
    return 0;
}

int cmlvadd(struct cmlvector * v1, struct cmlvector * v2) {
    /* Make sure both exist. */
    assert(v1 && v2);
    assert(v1->len == v2->len);
    for (int i = 0; i < v1->len; ++i) {
        v1->entries[i] += v2->entries[i];
    }
    return 0;
}

int cmlminit(struct cmlmatrix * m, int rows, int cols) {
    assert(m);
    m->m = rows;
    m->n = cols;
    m->entries = calloc(rows * cols, sizeof(float));
    assert(m->entries);
    return 0;
}

int cmlmfree(struct cmlmatrix * m) {
    assert(m);
    free(m->entries);
    return 0;
}