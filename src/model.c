#include "model.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

int cmlmodelinit(struct cmlmodel * model, int insize, int outsize, int layers) {
    assert(model);
    cmlninit(&model->net, insize, outsize, layers);
    cmlnrandinit(&model->net);
    return 0;
}


int cmlmodelfree(struct cmlmodel * model) {
    assert(model);
    cmlnfree(&model->net);
    return 0;
}

int cmlmodeladdtraining(struct cmlmodel * model, int train_no, struct cmlvector * tri, struct cmlvector * tro) {
    assert(model);
    assert(tri);
    assert(tro);
    /* Make sure the sizes line up. */
    for (int i = 0; i < train_no; ++i) {
        assert(tri[i].len == model->net.insize);
        assert(tro[i].len == model->net.outsize);
    }
    model->train_no = train_no;
    model->trains_in = tri;
    model->trains_out = tro;
    return 0;
}

int cmlmodeladdtesting(struct cmlmodel * model, int test_no, struct cmlvector * tei, struct cmlvector * teo) {
    assert(model);
    assert(tei);
    assert(teo);
    /* Make sure the sizes line up. */
    for (int i = 0; i < test_no; ++i) {
        assert(tei[i].len == model->net.insize);
        assert(teo[i].len == model->net.outsize);
    }
    model->test_no = test_no;
    model->tests_in = tei;
    model->tests_out = teo;
    return 0;
}

/**
 * Get the loss from a single vector pair.
 */
static float sqdistance(struct cmlvector * v1, struct cmlvector * v2) {
    assert(v1);
    assert(v2);
    assert(v1->len == v2->len);
    float total = 0;
    for (int i = 0; i < v1->len; ++i) {
        float diff = v1->entries[i] - v2->entries[i];
        total += diff * diff;
    }
    return total;
}

float cmlmodelgettestloss(struct cmlmodel * model) {
    assert(model);
    struct cmlvector output_buf;
    cmlvinit(&output_buf, model->net.insize);
    float loss = 0;
    for (int i = 0; i < model->test_no; ++i) {
        cmlnapp(&model->net, &model->tests_in[i], &output_buf);
        loss += sqdistance(&output_buf, &model->tests_out[i]);
        cmlvfree(&output_buf);
    }
    return loss / model->test_no;
}

float cmlmodelgettrainloss(struct cmlmodel * model) {
    assert(model);
    struct cmlvector output_buf;
    cmlvinit(&output_buf, model->net.insize);
    float loss = 0;
    for (int i = 0; i < model->train_no; ++i) {
        cmlnapp(&model->net, &model->trains_in[i], &output_buf);
        loss += sqdistance(&output_buf, &model->trains_out[i]);
        cmlvfree(&output_buf);
    }
    return loss / model->train_no;
}

/* Stores the partial derivatives, S, D, and D-hat values for a layer. */
/* <partials> is a list of all derivative lists, for each entry. */
struct pds {

    int len;

    struct cmlneuralnet * partials;

    struct cmlvector svals, dvals, dhats;

};

static int pdinit(struct pds * pd, struct cmlneuralnet * sinfo, int ln) {

    /* Size of layer's values */
    int s;
    if (ln == sinfo->layers)
        s = sinfo->outsize;
    else if (ln == 0)
        s = sinfo->insize;
    else
        s = sinfo->im_sizes[ln - 1];

    pd->len = s;

    pd->partials = calloc(s, sizeof(struct cmlneuralnet));
    assert(pd->partials);

    for (int i = 0; i < s; ++i)
        cmlninit(&(pd->partials[i]), sinfo->insize, sinfo->outsize, sinfo->layers);

    cmlvinit(&pd->svals, s);
    cmlvinit(&pd->dvals, s);
    cmlvinit(&pd->dhats, s);

    return 0;

}

static int pdfree(struct pds * pd) {
    for (int i = 0; i < pd->len; ++i)
        cmlnfree(&(pd->partials[i]));
    free(pd->partials);
    cmlvfree(&pd->svals);
    cmlvfree(&pd->dvals);
    cmlvfree(&pd->dhats);
    return 0;
}

/* Take one layer's pd-struct and calculate the next one's. ln is the layer whose entries we are calculating. */
static int recursepds(struct pds * cur, struct pds * next, struct cmlneuralnet * sinfo, int ln) {
    /* TODO - actual calculations here! */
    return 0;
}

/**
 * Get the partial derivatives from a single input and output set, and *add* them
 * to the currently stored value in pds.
 */
static int cmlgetpddp(struct cmlneuralnet * net, struct cmlvector * i, struct cmlvector * o, struct cmlneuralnet * pds) {

    /* Start with the 0th layer entries. */
    struct pds l, next;

    /* As outlined in the doc, S and D-hat are 0, while D is the input. */
    /* Also, all partial derivatives are zero here. */
    pdinit(&l, net, 0);
    memcpy(l.dvals.entries, i->entries, i->len * sizeof(float));

    /* Now, we recurse <layers> times to get the derivatives for the output layer. */
    for (int j = 1; j < net->layers + 1; ++j) {
        pdinit(&next, net, j);
        recursepds(&l, &next, net, j + 1);
        pdfree(&l);
        l = next;
    }

    /* TODO -- pd's are in <next>. Use the formula to find the pd's for loss. */

    /* Get the matrix pd's. */
    for (int j = 0; j < net->layers; ++j) {
        for (int k = 0; k < net->matrices[j].m * net->matrices[j].n; ++k) {
            for (int l = 0; l < net->outsize; ++l) {
                float partial = next.partials[l].matrices[j].entries[k];
                float diff = next.dvals.entries[l] - o->entries[l];
                pds->matrices[j].entries[k] += 2 * diff * partial;
            }
        }
    }

    /* Get the bias pd's. */
    for (int j = 0; j < net->layers; ++j) {
        for (int k = 0; k < net->biases[j].len; ++k) {
            for (int l = 0; l < net->outsize; ++l) {
                float partial = next.partials[l].biases[j].entries[k];
                float diff = next.dvals.entries[l] - o->entries[l];
                pds->biases[j].entries[k] += 2 * diff * partial;
            }
        }
    }

    pdfree(&next);

    return 0;

}

static int cmlgetpdfrompoint(struct cmlmodel * model, struct cmlneuralnet * pds, int i) {
    return cmlgetpddp(&model->net, &model->trains_in[i], &model->trains_out[i], pds);
}

int cmlmodellearn(struct cmlmodel * model, float learnspeed) {

    /* Every training datapoint will add its own weights to this. */
    struct cmlneuralnet tweaks;
    cmlninit(&tweaks, model->net.insize, model->net.outsize, model->net.layers);

    for (int i = 0; i < model->train_no; ++i) {
        cmlgetpdfrompoint(model, &tweaks, i);
    }

    /* Change all matrix entries */
    for (int i = 0; i < model->net.layers; ++i) {
        for (int p = 0; p < model->net.matrices[i].m * model->net.matrices[i].n; ++p) {
            model->net.matrices[i].entries[p] -= tweaks.matrices[i].entries[p] * learnspeed;
        }
    }

    /* Change all bias entries */
    for (int i = 0; i < model->net.layers - 1; ++i) {

        int s;
        if (i == model->net.layers - 1)
            s = model->net.outsize;
        else
            s = model->net.im_sizes[i];
        
        for (int p = 0; p < s; ++p) {
            model->net.biases[i].entries[p] -= tweaks.biases[i].entries[p] * learnspeed;
        }
    }

    return 0;

}
