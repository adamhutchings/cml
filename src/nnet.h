/**
 * The API for a neural network. Includes code for initializing and freeing a
 * neural network, and running a vector through the model.
 */

#ifndef CML_NNET_H
#define CML_NNET_H

#include "mvmath.h"

struct cmlneuralnet {

    /* This is the number of matrices that the net has. */
    int layers;

    /* A list of <layers> cmlmatrix pointers. */
    struct cmlmatrix ** matrices;

    /* A list of <layers> bias vectors. */
    struct cmlvector * biases;

    /* A list of <layers-1> intermediate sizes. Like if the input is 10 long and
    the output is 3, then the list being 5, 4 would indicate matrices of size
    5x10, 4x5, and 3x4. */
    int * im_sizes;

};

/**
 * Construct a neural net with the specified input size, output size, and number
 * of layers in the net.
 */
int cmlninit(struct cmlneuralnet * net, int insize, int outsize, int layers);

/**
 * Destroy the memory usage in the neural net.
 */
int cmlnfree(struct cmlneuralnet * net);

#endif /* CML_NNET_H */
