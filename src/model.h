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

    int train_no;
    struct cmlvector * trains_in;
    struct cmlvector * trains_out;

    int test_no;
    struct cmlvector * tests_in;
    struct cmlvector * tests_out;

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
int cmlmodellearn(struct cmlmodel * model, float learnspeed);

int cmlmodeltrain(struct cmlmodel * model);

#endif /* CML_MODEL_H */
