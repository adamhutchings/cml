#include <assert.h>
#include <math.h>
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

    float trainloss, testloss;
    float oldtr, oldte;

    /* For Iris, our goal loss is less than 0.04. */

    printf("%s\n", "Testing on iris dataset ...");

    time_t start, end;

    start = time(&start);

    for (int i = 0; ; ++i) {
        trainloss = cmlmodelgettrainloss(&model);
        testloss = cmlmodelgettestloss(&model);
        cmlmodellearn(&model, 0.00001 * trainloss);
        if (i % 100 == 0) {
            if (i % 1000 == 0)
                printf("After %d rounds: training loss: %f, testing loss: %f.\n", i, trainloss, testloss);
            if (i > 0 && trainloss > oldtr) {
                end = time(&end);
                printf("Finished after %d rounds in %.2ld seconds.\n", i, end - start);
                printf("Training loss: %f, testing loss: %f.\n", trainloss, testloss);
                break;
            }
            oldtr = trainloss;
            oldte = testloss;
        }
    }

    /* See the actual outputs. */

    struct cmlvector outbuf;

    for (int i = 0; i < model.test_no; ++i) {
        cmlnapp(&model.net, &model.tests_in[i], &outbuf);
        printf("Expected value: %.2f, actual value: %.2f.\n", model.tests_out[i].entries[0], outbuf.entries[0]);
        cmlvfree(&outbuf);
    }

    cmlmodelfree(&model);

    cmlfreedata(tri, tro, trainno);
    cmlfreedata(tei, teo, testno);

    return 0;

}

int main(int argc, char ** argv) {
    modeltest();
    return 0;
}
