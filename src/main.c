#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "iris-reader.h"
#include "nnet.h"
#include "model.h"

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

    float trainloss = cmlmodelgettrainloss(&model);
    float testloss = cmlmodelgettestloss(&model);

    printf("Before: training loss: %f, testing loss: %f.\n", trainloss, testloss);

    /* Iterate one learning cycle. */
    cmlmodellearn(&model, 0.001);

    trainloss = cmlmodelgettrainloss(&model);
    testloss = cmlmodelgettestloss(&model);

    printf("After: training loss: %f, testing loss: %f.\n", trainloss, testloss);

    cmlmodelfree(&model);

    cmlfreedata(tri, tro, trainno);
    cmlfreedata(tei, teo, testno);

    return 0;

}

int main(int argc, char ** argv) {
    modeltest();
    return 0;
}
