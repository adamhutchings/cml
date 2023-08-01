#ifndef CML_MNIST_READER_H
#define CML_MNIST_READER_H

#include "mvmath.h"

/**
 * Defines capabilities to read in a data file in the format of the iris data
 * and produce the list of input and output vectors.
 */

int cmlreadmnist(const char * filepath, struct cmlvector ** ins, struct cmlvector ** outs, int * size);

int cmlfreemnist(struct cmlvector * ins, struct cmlvector * outs, int no);

#define MNIST_INPUT_SIZE (28 * 28)
#define MNIST_OUTPUT_SIZE 10

#endif /* CML_MNIST_READER_H */
