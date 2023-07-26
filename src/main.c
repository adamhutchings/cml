#include <assert.h>
#include <stdio.h>

#include "nnet.h"

/**
 * Make sure that the intermediate sizes are as expected.
 */
int nntest() {
    struct cmlneuralnet net;
    cmlninit(&net, 100, 1, 3);
    printf("Input size: %d\n", net.insize);
    printf("1st intermediate: %d\n", net.im_sizes[0]);
    printf("2nd intermediate: %d\n", net.im_sizes[1]);
    printf("Output size: %d\n", net.outsize);
    /* Make sure the matrices are of the right size */
    printf("1st layer input  size: %d\n", net.matrices[0].n);
    printf("1st layer output size: %d\n", net.matrices[0].m);
    printf("2nd layer input  size: %d\n", net.matrices[1].n);
    printf("2nd layer output size: %d\n", net.matrices[1].m);
    printf("3rd layer input  size: %d\n", net.matrices[2].n);
    printf("3rd layer output size: %d\n", net.matrices[2].m);
    assert(net.matrices[0].n == net.insize);
    assert(net.matrices[0].m == net.im_sizes[0]);
    assert(net.matrices[1].n == net.im_sizes[0]);
    assert(net.matrices[1].m == net.im_sizes[1]);
    assert(net.matrices[2].n == net.im_sizes[1]);
    assert(net.matrices[2].m == net.outsize);
    /* Let's make sure we can actually multiply something. */
    struct cmlvector in, out;
    cmlvinit(&in, 100);
    cmlnapp(&net, &in, &out);
    printf("Output size: %d\n", out.len);
    /* Output should be 0 because every entry in the net is 0. */
    printf("Output: %f\n", out.entries[0]);
    cmlvfree(&in);
    cmlvfree(&out);
    cmlnfree(&net);
    return 0;
}

int main(int argc, char ** argv) {
    printf("%s\n", "Hello, World!");
    nntest();
    return 0;
}
