#ifndef CML_IRIS_READER_H
#define CML_IRIS_READER_H

#include "mvmath.h"

/**
 * Defines capabilities to read in a data file in the format of the iris data
 * and produce the list of input and output vectors.
 */

int cmlreadiris(const char * filepath, struct cmlvector ** ins, struct cmlvector ** outs, int * size);

int cmlfreedata(struct cmlvector * ins, struct cmlvector * outs, int no);

#endif /* CML_IRIS_READER_H */
