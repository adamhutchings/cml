#include "mvmath.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

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

float cmlmentry(struct cmlmatrix * m, int i, int j) {
    assert(
        (i >= 0)
    &&  (j >= 0)
    &&  (i < m->m)
    &&  (j < m->n)
    );
    return m->entries[i * m->n + j];
}

int cmlmsentry(struct cmlmatrix * m, int i, int j, float v) {
    assert(
        (i >= 0)
    &&  (j >= 0)
    &&  (i < m->m)
    &&  (j < m->n)
    );
    m->entries[i * m->n + j] += v;
    return 0;
}

int cmlmul(struct cmlmatrix * m, struct cmlvector * v1, struct cmlvector * v2) {

    /* The checks are that v1 has as many entries as m has columns, and that v2
    has as many entries as m has rows. */

    assert(v1->len == m->n);
    assert(v2->len == m->m);

    /* Ready the field. */
    memset(v2->entries, 0, v2->len * sizeof(float));

    for (int i = 0; i < m->m; ++i) {
        /* The ith entry in the output vector is the sum over all j of the jth
        entry in the input times the (i, j)th entry in the matrix. */
        for (int j = 0; j < m->n; ++j) {
            v2->entries[i] += v1->entries[j] * cmlmentry(m, i, j);
        }
    }

    return 0;

}
