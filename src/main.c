#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "iris-reader.h"
#include "nnet.h"
#include "model.h"

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

    int trainno, testno;
    struct cmlvector * tri, * tro, * tei, * teo;
    struct cmlmodel model;

    cmlmodelinit(&model, 4, 1, 1);

    cmlreadiris("data/iris/iristraining.csv", &tri, &tro, &trainno);
    cmlreadiris("data/iris/iristesting.csv", &tei, &teo, &testno);

    cmlmodeladdtraining(&model, trainno, tri, tro);
    cmlmodeladdtesting(&model, testno, tei, teo);

    /* Just make sure the loss calculation is going alright. */
    float trainloss = cmlmodelgettrainloss(&model);
    float testloss = cmlmodelgettestloss(&model);

    printf("Training loss: %f, testing loss: %f.\n", trainloss, testloss);

    cmlmodelfree(&model);

    cmlfreedata(tri, tro, trainno);
    cmlfreedata(tei, teo, testno);

    return 0;

}

int main(int argc, char ** argv) {
    modeltest();
    return 0;
}
