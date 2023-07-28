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

    printf("%s\n", "Testing on iris dataset ...");

    time_t start, end;

    start = time(&start);

    for (int i = 0; ; ++i) {
        trainloss = cmlmodelgettrainloss(&model);
        testloss = cmlmodelgettestloss(&model);
        if (i % 1000 == 0)
            printf("After %d rounds: training loss: %f, testing loss: %f.\n", i, trainloss, testloss);
        cmlmodellearn(&model, 0.000001 * 1.0f / trainloss);
        if (i > 100 && trainloss > oldtr) {
            end = time(&end);
            printf("Finished after %d rounds in %.2ld seconds.\n", i, end - start);
            printf("Training loss: %f, testing loss: %f.\n", trainloss, testloss);
            break;
        }
        oldtr = trainloss;
        oldte = testloss;
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
