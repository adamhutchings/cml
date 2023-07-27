#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "iris-reader.h"
#include "nnet.h"

/**
 * Make sure that the intermediate sizes are as expected.
 */
int nntest() {
    struct cmlneuralnet net;
    cmlninit(&net, 100, 1, 3);
    struct cmlvector in, out;
    cmlvinit(&in, 100);
    cmlnrandinit(&net);
    /* Test 1 million applications. */
    time_t start, end;
    time(&start);
    int trials = 1000000;
    int pct = trials / 100;
    for (int i = 0; i < trials; ++i) {
        if (i % pct == pct - 1) {
            printf("%d%% of the way done.\n", (i + 1) / pct);
        }
        cmlnapp(&net, &in, &out);
    }
    time(&end);
    double avgtime = (end - start) / (double) trials;
    printf("Average time to run neural net is %f seconds.\n", avgtime);
    int totalentries = 0;
    for (int i = 0; i < net.layers; ++i) {
        totalentries += net.matrices[i].m * net.matrices[i].n;
        totalentries += net.biases[i].len;
    }
    printf("Total parameters in the net: %d\n", totalentries);
    printf("Time to complete one learning cycle using epsilon method: %f seconds.\n", avgtime * (double) totalentries);
    cmlvfree(&in);
    cmlvfree(&out);
    cmlnfree(&net);
    return 0;
}

/**
 * Test the basics of a model.
 */
int modeltest() {
    int no;
    struct cmlvector * ins;
    struct cmlvector * outs;
    cmlreadiris("data/iris/iristesting.csv", &ins, &outs, &no);
    /* Making sure the first row of data is read right. */
    printf(
        "%.1f %.1f %.1f %.1f -> %.1f\n",
        ins[0].entries[0],
        ins[0].entries[1],
        ins[0].entries[2],
        ins[0].entries[3],
        outs[0].entries[0]
    );
    printf("%d\n", no);
    cmlfreedata(ins, outs, no);
    return 0;
}

int main(int argc, char ** argv) {
    modeltest();
    return 0;
}
