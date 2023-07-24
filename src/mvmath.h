#ifndef CML_MVMATH_H
#define CML_MVMATH_H

/**
 * Definitions for matrix and vector math. That is, adding vectors, multiplying
 * them by matrices, and initializing them conveniently.
 */

/**
 * The definition of a vector -- a bunch of float entries and a length.
 */
struct cmlvector {
    float * entries;
    int len;
};

int cmlvinit(struct cmlvector * v, int size);
int cmlvfree(struct cmlvector * v);

/* This adds each entry in v2 to v1. */
int cmlvadd (struct cmlvector * v1, struct cmlvector * v2);

/**
 * A matrix with m rows and n columns.
 */
struct cmlmatrix {
    float * entries;
    int m, n;
};

int cmlminit(struct cmlmatrix * m, int rows, int cols);
int cmlmfree(struct cmlmatrix * m);

#endif /* CML_MVMATH_H */
