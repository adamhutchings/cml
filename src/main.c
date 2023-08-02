#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "iris-reader.h"
#include "nnet.h"
#include "mnist-reader.h"
#include "model.h"

/**
 * Test the IRIS dataset.
 */
int iristest() {

    int trainno, testno;
    struct cmlvector * tri, * tro, * tei, * teo;
    struct cmlmodel model;

    cmlmodelinit(&model, 4, 1, 1);

    cmlreadiris("data/iris/iristraining.csv", &tri, &tro, &trainno);
    cmlreadiris("data/iris/iristesting.csv", &tei, &teo, &testno);

    cmlmodeladdtraining(&model, trainno, tri, tro);
    cmlmodeladdtesting(&model, testno, tei, teo);

    struct cmlhyperparams params;
    params.iterations = 0;
    params.learning_speed = 0.0001f;
    params.sw = 0;
    params.error_threshold = 0.04;
    params.training_max = 0;
    params.status_rarity = 100;

    /* For Iris, our goal loss is less than the error threshold. */
    cmlmodeltrain(&model, &params);

    printf("%s\n", "Testing on iris dataset ...");

    /* See the actual outputs. */

    struct cmlvector outbuf;

    for (int i = 0; i < model.test_no; ++i) {
        cmlnapp(&model.net, &model.tests_in[i], &outbuf);
        printf("Expected value: %.2f, actual value: %.2f.\n", model.tests_out[i].entries[0], outbuf.entries[0]);
        cmlvfree(&outbuf);
    }

    cmlmodelfree(&model);

    cmlfreeiris(tri, tro, trainno);
    cmlfreeiris(tei, teo, testno);

    return 0;

}

/* Get the largest value in an array. */
int array_highest(float * arr, int sz) {
    int h = -1;
    float v = -1000;
    for (int i = 0; i < sz; ++i) {
        if (arr[i] > v) {
            v = arr[i];
            h = i;
        }
    }
    return h;
}

/* Get the accuracy from MNIST data by seeing how well it predicts a number. */
float mnist_accuracy(struct cmlmodel * model) {
    int correct = 0;
    struct cmlvector outbuf;
    for (int i = 0; i < model->test_no; ++i) {
        cmlnapp(&model->net, &model->tests_in[i], &outbuf);
        if (array_highest(outbuf.entries, MNIST_OUTPUT_SIZE) == array_highest(model->tests_out[i].entries, MNIST_OUTPUT_SIZE))
            ++correct;
    }
    return (float) correct / model->test_no;
}

int mnisttest() {

    int trainno, testno;
    struct cmlvector * tri, * tro, * tei, * teo;
    struct cmlmodel model;

    printf("%s\n", "Initializing model ...");

    cmlmodelinit(&model, MNIST_INPUT_SIZE, MNIST_OUTPUT_SIZE, 2);

    printf("%s\n", "Reading training data ...");
    cmlreadmnist("data/mnist/mnisttraining.csv", &tri, &tro, &trainno);
    printf("%s\n", "Reading testing data ...");
    cmlreadmnist("data/mnist/mnisttesting.csv", &tei, &teo, &testno);

    cmlmodeladdtraining(&model, trainno, tri, tro);
    cmlmodeladdtesting(&model, testno, tei, teo);

    struct cmlhyperparams params;
    params.error_threshold = 0;
    params.iterations = 0;
    params.learning_speed = 0.0001;
    params.sw = 1;
    params.sw_size = 100;
    params.training_max = 10;
    params.status_rarity = 1;

    printf("Starting error: %f\n", cmlmodelgettestloss(&model));
    printf("Starting accuracy: %.3f%%\n", mnist_accuracy(&model) * 100);

    cmlmodeltrain(&model, &params);

    printf("Ending error: %f\n", cmlmodelgettestloss(&model));
    printf("Ending accuracy: %.3f%%\n", mnist_accuracy(&model) * 100);

    cmlmodelfree(&model);

    cmlfreemnist(tri, tro, trainno);
    cmlfreemnist(tei, teo, testno);

    return 0;

}

int main(int argc, char ** argv) {
    iristest();
    /* mnisttest(); */
    return 0;
}
