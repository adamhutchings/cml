#include "nnet.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

int cmlninit(struct cmlneuralnet * net, int insize, int outsize, int layers) {

    assert(net);
    assert(insize > 0);
    assert(outsize > 0);
    assert(layers > 0);

    net->layers = layers;
    net->insize = insize;
    net->outsize = outsize;

    /* The layer sizes will form as close to a geometric progression as we can.
    For example, if insize is 256 and outsize is 4, with 3 layers we have 2
    internal sizes, which will be chosen to be 64 and 16. */
    net->im_sizes = calloc(layers - 1, sizeof(int));
    for (int i = 1; i < layers; ++i) {
        net->im_sizes[i - 1] = insize * pow(
            (double) outsize / insize, (float) i / layers
        );
    }

    /* Make matrices of size insize -> im_sizes[0] -> im_sizes[1], ...
    im_sizes[layers - 3] -> im_sizes[layers - 2] -> outsize . */
    /* Just to make things neat, we'll deal with the small-size cases separately
    - 1 layer (one matrix), 2 layers (one internal size). */
    net->matrices = calloc(layers, sizeof (struct cmlmatrix));
    if (layers == 1) {
        cmlminit(&(net->matrices[0]), insize, outsize);
    } else if (layers == 2) {
        int is = net->im_sizes[0];
        cmlminit(&(net->matrices[0]), insize, is);
        cmlminit(&(net->matrices[1]), is, outsize);
    } else {
        cmlminit(&(net->matrices[0]), insize, net->im_sizes[0]);
        cmlminit(&(net->matrices[layers - 1]), net->im_sizes[layers - 2], outsize);
        for (int i = 0; i < layers - 2; ++i) {
            cmlminit(&(net->matrices[i + 1]), net->im_sizes[i], net->im_sizes[i + 1]);
        }
    }

    net->biases = calloc(layers, sizeof (struct cmlvector));
    for (int i = 0; i < layers - 1; ++i) {
        cmlvinit(&(net->biases[i]), net->im_sizes[i]);
    }
    cmlvinit(&(net->biases[layers - 1]), net->outsize);

    return 0;

}

int cmlnfree(struct cmlneuralnet * net) {

    assert(net);

    /* List of things to free:
    Every matrix, then the matrix list itself
    Every bias vector, then the bias vector list itself
    The list of intermediate sizes */

    for (int i = 0; i < net->layers; ++i) {
        cmlmfree(&(net->matrices[i]));
    }
    free(net->matrices);

    for (int i = 0; i < net->layers; ++i) {
        cmlvfree(&(net->biases[i]));
    }
    free(net->biases);

    free(net->im_sizes);

    return 0;

}

/* Random float between -1 and 1. */
float randfloat() {
    float f = ((float) rand()) / ((float) RAND_MAX);
    /* Move it from 0 ... 1 to -1 ... 1. */
    return f + f - 1.0f;
}

/* Randomly seed every matrix and bias entry between -1 and 1. */
int cmlnrandinit(struct cmlneuralnet * net) {

    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < net->layers; ++i) {
        /* Matrices */
        for (int j = 0; j < net->matrices[i].m * net->matrices[i].n; ++j) {
            net->matrices[i].entries[j] = randfloat();
        }
        for (int j = 0; j < net->biases[i].len; ++j) {
            net->biases[i].entries[j] = randfloat();
        }
    }

    return 0;

}

/* Activation function. */
static float sigmoid(float input) {
    return 1.0f / (1 + exp(-input));
}

static int applyactivation(struct cmlvector * v) {
    for (int i = 0; i < v->len; ++i) {
        v->entries[i] = sigmoid(v->entries[i]);
    }
    return 0;
}

int cmlnapp(struct cmlneuralnet * net, struct cmlvector * in, struct cmlvector * out) {

    assert(net && in && out);
    assert(net->insize == in->len);

    /* Just put through each layer (matrix and biases). */
    struct cmlvector * lin, * lout;

    cmlvinit(out, net->outsize);

    /* If only 1 layer: */
    if (net->layers == 1) {
        cmlmul(&(net->matrices[0]), in, out);
        cmlvadd(out, &(net->biases[0]));
        return 0;
    }

    /* We go through each net and set up the input and output accordingly,
    feeding each output in as the input of the next layer. */
    for (int i = 0; i < net->layers; ++i) {
        if (i == 0)
            lin = in;
        else
            lin = lout;
        if (i == net->layers - 1) {
            lout = out;
        } else {
            lout = malloc(sizeof (struct cmlvector));
            cmlvinit(lout, net->im_sizes[i]);
        }
        cmlmul(&(net->matrices[i]), lin, lout);
        cmlvadd(lout, &(net->biases[i]));
        applyactivation(lout);
        if (i != 0) {
            cmlvfree(lin);
            free(lin);
        }
    }

    return 0;

}
