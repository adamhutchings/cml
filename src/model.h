#ifndef CML_MODEL_H
#define CML_MODEL_H

#include "nnet.h"
#include "mvmath.h"

/**
 * Defines a model, with a neural net, trainig dataset, expected output, and
 * testing data. 
 */
struct cmlmodel {

    struct cmlneuralnet net;
    struct cmlneuralnet lasts;

    int train_no;
    struct cmlvector * trains_in;
    struct cmlvector * trains_out;

    int test_no;
    struct cmlvector * tests_in;
    struct cmlvector * tests_out;

};

/**
 * Defines learning parameters and current training information.
 */
struct cmlhyperparams {

    /* How many training cycles we've gone through. */
    int iterations;

    /* Are we doing a spinning window? */
    int sw;
    /* For spinning-window -- how large the window is. */
    int sw_size;

    /* Baseline speed to adjust learning at. */
    float learning_speed;

    /* How far down we need to get. */
    float error_threshold;

};

int cmlmodelinit(struct cmlmodel * model, int insize, int outsize, int layers);

int cmlmodelfree(struct cmlmodel * model);

int cmlmodeladdtraining(struct cmlmodel * model, int train_no, struct cmlvector * tri, struct cmlvector * tro);

int cmlmodeladdtesting(struct cmlmodel * model, int test_no, struct cmlvector * tei, struct cmlvector * teo);

float cmlmodelgettestloss(struct cmlmodel * model);

float cmlmodelgettrainloss(struct cmlmodel * model);

/**
 * This is the real meat. Train the neural network one round.
 */
int cmlmodellearn(struct cmlmodel * model, float learnspeed, float inertia);

int cmlmodeltrain(struct cmlmodel * model, struct cmlhyperparams * params);

#endif /* CML_MODEL_H */
