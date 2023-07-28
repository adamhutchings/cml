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

    /* For Iris, our goal loss is less than 0.04. */
    cmlmodeltrain(&model, 0.04);

    printf("%s\n", "Testing on iris dataset ...");

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
