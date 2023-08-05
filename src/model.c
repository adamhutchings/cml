#include "model.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mvmath.h"

int cmlmodelinit(struct cmlmodel * model, int insize, int outsize, int layers) {
    assert(model);
    cmlninit(&model->net, insize, outsize, layers);
    cmlninit(&model->lasts, insize, outsize, layers);
    return 0;
}

int cmlmodelsetlayersize(struct cmlmodel * model, int lno, int size) {
    cmlnsetlayersize(&model->net, lno, size);
    return 0;
}

int cmlmodelmakenets(struct cmlmodel * model) {
    cmlnmakewb(&model->net);
    cmlnrandinit(&model->net, 0.1f);
    cmlncopysizes(&model->lasts, &model->net);
    cmlnmakewb(&model->lasts);
    return 0;
}

int cmlmodelfree(struct cmlmodel * model) {
    assert(model);
    cmlnfree(&model->net);
    cmlnfree(&model->lasts);
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

static float cmlmodelgettrainloss_ri(struct cmlmodel * model, int ss, int es) {
    assert(model);
    struct cmlvector output_buf;
    cmlvinit(&output_buf, model->net.insize);
    float loss = 0;
    for (int j = ss; j < es; ++j) {
        int i = j % model->train_no;
        cmlnapp(&model->net, &model->trains_in[i], &output_buf);
        loss += sqdistance(&output_buf, &model->trains_out[i]);
        cmlvfree(&output_buf);
    }
    return loss / (es - ss);
}

static float cmlmodelgettestloss_ri(struct cmlmodel * model, int ss, int es) {
    assert(model);
    struct cmlvector output_buf;
    cmlvinit(&output_buf, model->net.insize);
    float loss = 0;
    for (int j = ss; j < es; ++j) {
        int i = j % model->test_no;
        cmlnapp(&model->net, &model->tests_in[i], &output_buf);
        loss += sqdistance(&output_buf, &model->tests_out[i]);
        cmlvfree(&output_buf);
    }
    return loss / (es - ss);
}

struct lossparams {
    struct cmlmodel * model;
    int ss, es;
    float * storeloc;
};

static void * cmgtrrp(struct lossparams * lp) {
    *(lp->storeloc) = cmlmodelgettrainloss_ri(lp->model, lp->ss, lp->es);
    return NULL;
}

static void * cmgterp(struct lossparams * lp) {
    *(lp->storeloc) = cmlmodelgettestloss_ri(lp->model, lp->ss, lp->es);
    return NULL;
}

/* Test is either 0 or 1. If 0, we do it on the training data. */
static float cmlmodelgettloss_r(struct cmlmodel * model, int ss, int es, int test) {

    unsigned t = model->threads;

    /* Where to store the results from each computation. */
    float * results = calloc(t, sizeof(float));

    /* All of the threads we're making. */
    pthread_t * threads = calloc(t, sizeof(pthread_t));

    /* The sets of arguments to the function. */
    struct lossparams * lps = calloc(t, sizeof(struct lossparams));
    for (int i = 0; i < t; ++i) {
        lps[i].model = model;
        lps[i].ss = ss + ((i + 0) * (es - ss)) / t;
        lps[i].es = ss + ((i + 1) * (es - ss)) / t;
        lps[i].storeloc = &(results[i]);
    }

    void * func = test ? cmgterp : cmgtrrp;

    for (int i = 0; i < t; ++i) {
        pthread_create(&threads[i], NULL, func, &(lps[i]));
    }

    for (int i = 0; i < t; ++i) {
        pthread_join(threads[i], NULL);
    }

    float r = 0.f;

    for (int i = 0; i < t; ++i) {
        r += results[i];
    }

    free(results);
    free(threads);
    free(lps);

    return r / t;

}

static float cmlmodelgettestloss_r(struct cmlmodel * model, int ss, int es) {
    return cmlmodelgettloss_r(model, ss, es, 1);
}

static float cmlmodelgettrainloss_r(struct cmlmodel * model, int ss, int es) {
    return cmlmodelgettloss_r(model, ss, es, 0);
}

float cmlmodelgettestloss(struct cmlmodel * model) {
    return cmlmodelgettestloss_r(model, 0, model->test_no);
}

float cmlmodelgettrainloss(struct cmlmodel * model) {
    return cmlmodelgettrainloss_r(model, 0, model->train_no);
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

    for (int i = 0; i < s; ++i) {
        cmlninit(&(pd->partials[i]), sinfo->insize, sinfo->outsize, sinfo->layers);
        cmlncopysizes(&(pd->partials[i]), sinfo);
        cmlnmakewb(&(pd->partials[i]));
    }

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

    /* Calculate S, D, and D-hat values. */

    assert(next->svals.len == next->dvals.len);
    assert(next->svals.len == next->dhats.len);

    int dk = next->svals.len;
    int layers = sinfo->layers;

    /* S values */
    cmlmul(&(sinfo->matrices[ln]), &cur->dvals, &next->svals);
    cmlvadd(&next->svals, &sinfo->biases[ln]);

    /* D values */
    for (int i = 0; i < dk; ++i) {
        next->dvals.entries[i] = activation(next->svals.entries[i]);
    }

    /* D-hat values */
    for (int i = 0; i < dk; ++i) {
        next->dhats.entries[i] = activation_derivative(next->svals.entries[i]);
    }

    int lsize;

    if (ln == sinfo->layers)
        lsize = sinfo->outsize;
    else if (ln == 0)
        lsize = sinfo->insize;
    else
        lsize = sinfo->im_sizes[ln - 1];

    /* Calculate matrix entry pd's */

    /* The derivative of D_k,x with respect to ... */
    for (int x = 0; x < dk; ++x) {

        /* Higher-layer matrices have derivatives of zero */
        for (int kp = ln + 1; kp < layers; ++kp) {
            for (int p = 0; p < sinfo->matrices[kp].m * sinfo->matrices[kp].n; ++p) {
                next->partials[x].matrices[kp].entries[p] = 0;
            }
        }

        /* A_k,m,n */
        int maxm = sinfo->matrices[ln].m, maxn = sinfo->matrices[ln].n;
        for (int m = 0; m < maxm; ++m) {
            for (int n = 0; n < maxn; ++n) {
                int pos = m * maxn + n;
                if (x == m) {
                    next->partials[x].matrices[ln].entries[pos] = next->dhats.entries[x] * cur->dvals.entries[n];
                } else {
                    next->partials[x].matrices[ln].entries[pos] = 0;
                }
            }
        }

        /* Lower-level uses recursion. */
        for (int kp = 0; kp < ln; ++kp) {
            for (int m = 0; m < sinfo->matrices[kp].m; ++m) {
                for (int n = 0; n < sinfo->matrices[kp].n; ++n) {
                    int pos = m * maxn + n;
                    float sum = 0;
                    for (int j = 0; j < lsize; ++j) {
                        sum += sinfo->matrices[ln].entries[x * lsize + j] * cur->partials[j].matrices[kp].entries[pos];
                    }
                    next->partials[x].matrices[kp].entries[pos] = sum * next->dhats.entries[x];
                }
            }
        }

    }

    /* Calculate bias pd's */
    for (int x = 0; x < dk; ++x) {

        /* Higher-level biases have derivatives of zero */
        for (int kp = ln + 1; kp < layers; ++kp) {
            for (int p = 0; p < sinfo->biases[kp].len; ++p) {
                next->partials[x].biases[kp].entries[p] = 0;
            }
        }

        /* Current-level biases */
        for (int p = 0; p < sinfo->biases[ln].len; ++p) {
            next->partials[x].biases[ln].entries[p] = 0;
        }
        next->partials[x].biases[ln].entries[x] = next->dhats.entries[x];

        /* Previous biases */
        /* Lower-level uses recursion. */
        for (int kp = 0; kp < ln; ++kp) {
            for (int p = 0; p < sinfo->biases[kp].len; ++p) {
                float sum = 0;
                for (int j = 0; j < lsize; ++j) {
                    sum += sinfo->matrices[ln].entries[x * lsize + j] * cur->partials[j].biases[kp].entries[p];
                }
                next->partials[x].biases[kp].entries[p] = sum * next->dhats.entries[x];
            }
        }

    }

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
        recursepds(&l, &next, net, j - 1);
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

struct pdparams {
    struct cmlmodel * model;
    struct cmlneuralnet * pds;
    int ss;
    int es;
};

/* Get the pds from a range of datapoints. */
static void * cmlgetpdfromrange(struct pdparams * pdp) {
    for (int i = pdp->ss; i < pdp->es; ++i)
        cmlgetpdfrompoint(pdp->model, pdp->pds, i % pdp->model->train_no);
    return NULL;
}

static int cmlmodelmaketweak(struct cmlmodel * model, float tweak_amount, struct cmlneuralnet * tweaks, float learnspeed) {

    /* Change all matrix entries */
    for (int i = 0; i < model->net.layers; ++i) {
        for (int p = 0; p < model->net.matrices[i].m * model->net.matrices[i].n; ++p) {
            float n = -tweaks->matrices[i].entries[p] * learnspeed;
            model->net.matrices[i].entries[p] += n;
            model->lasts.matrices[i].entries[p] = n;
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
            float n = -tweaks->biases[i].entries[p] * learnspeed;
            model->net.biases[i].entries[p] += n;
            model->lasts.biases[i].entries[p] = n;
        }
    }

    return 0;

}

/**
 * Add regularizations. 
 */
static int cmlmodeladdreg(struct cmlmodel * model, struct cmlneuralnet * tweaks, float lambda) {

    /* Change all matrix entries */
    for (int i = 0; i < model->net.layers; ++i) {
        for (int p = 0; p < model->net.matrices[i].m * model->net.matrices[i].n; ++p) {
            tweaks->matrices[i].entries[p] -= 2 * model->net.matrices[i].entries[p] * lambda;
        }
    }

    /* No bias change. */
    return 0;

}

/**
 * Returns how many steps it took to improve.
*/
float cmlmodellearn(struct cmlmodel * model, float learnspeed, int ss, int es, float lambda) {

    int t = model->threads;

    struct cmlneuralnet * tweaksets = calloc(t, sizeof(struct cmlneuralnet));
    for (int i = 0; i < t; ++i) {
        cmlninit(&tweaksets[i], model->net.insize, model->net.outsize, model->net.layers);
        cmlncopysizes(&tweaksets[i], &model->net);
        cmlnmakewb(&tweaksets[i]);
    }

    struct pdparams * paramsets = calloc(t, sizeof(struct pdparams));
    for (int i = 0; i < t; ++i) {
        paramsets[i].model = model;
        paramsets[i].pds = &tweaksets[i];
        paramsets[i].ss = ss + ((i + 0) * (es - ss)) / t;
        paramsets[i].es = ss + ((i + 1) * (es - ss)) / t;
    }

    pthread_t * threads = calloc(t, sizeof(pthread_t));

    for (int i = 0; i < t; ++i) {
        pthread_create(&threads[i], NULL, (void *) cmlgetpdfromrange, &paramsets[i]);
    }

    for (int i = 0; i < t; ++i) {
        pthread_join(threads[i], NULL);
    }

    struct cmlneuralnet tweaks;
    cmlninit(&tweaks, model->net.insize, model->net.outsize, model->net.layers);
    cmlncopysizes(&tweaks, &model->net);
    cmlnmakewb(&tweaks);

    for (int i1 = 0; i1 < t; ++i1) {
        /* Change all matrix entries */
        for (int i = 0; i < model->net.layers; ++i) {
            for (int p = 0; p < model->net.matrices[i].m * model->net.matrices[i].n; ++p) {
                tweaks.matrices[i].entries[p] += tweaksets[i1].matrices[i].entries[p];
            }
        }
        for (int i = 0; i < model->net.layers - 1; ++i) {
            int s;
            if (i == model->net.layers - 1)
                s = model->net.outsize;
            else
                s = model->net.im_sizes[i];
            for (int p = 0; p < s; ++p) {
                tweaks.biases[i].entries[p] += tweaksets[i1].biases[i].entries[p];
            }
        }
    }

    for (int i = 0; i < t; ++i) {
        cmlnfree(&tweaksets[i]);
    }
    free(tweaksets);
    free(paramsets);
    free(threads);

    cmlmodeladdreg(model, &tweaks, lambda);

    float rtval = 0;

    /* Make tweaks until we stop improving, and then undo the last tweak. */
    float loss = cmlmodelgettrainloss_r(model, ss, es), loss2;
    float pr = 1 / (es - ss);
    for (int i = 0; ; ++i) {
        /* printf("%d\n", i); */
        cmlmodelmaketweak(model, pr, &tweaks, learnspeed);
        loss2 = cmlmodelgettrainloss_r(model, ss, es);
        if (loss2 > loss) {
            cmlmodelmaketweak(model, -pr, &tweaks, learnspeed);
            if (i == 0)
                rtval = 0.5f;
            else
                rtval = 1.f;
            break;
        }
        loss = loss2;
    }

    cmlnfree(&tweaks);
    return rtval;

}

int cmlmodeltrain(struct cmlmodel * model, struct cmlhyperparams * params) {

    float trainloss, testloss;
    float oldtr, oldte;

    time_t start, end;

    start = time(&start);

    params->iterations = 0;

    int i = 0, ss, es;

    float sp = params->learning_speed;

    while (params->training_max != 0 ? i < params->training_max : 1) {

        if (params->sw) {
            ss = i * params->sw_size;
            es = ss + params->sw_size;
        } else {
            ss = 0;
            es = model->train_no;
        }

        trainloss = cmlmodelgettrainloss_r(model, ss, es);
        testloss  = cmlmodelgettestloss_r (model, ss, es);

        sp *= cmlmodellearn(model, sp, ss, es, params->lambda);
        ++i;

        if (i % params->status_rarity == 0)
            printf("After %d rounds: training error %.6f, testing error %.6f.\n", i, trainloss, testloss);

        oldtr = trainloss, oldte = testloss;

        /* if (i > 0 && (oldtr - trainloss) / trainloss < 0.001) {
            printf("Stopped improving after %d rounds.\n", i);
            break;
        } */

        if (trainloss < params->error_threshold) {
            printf("Passed below error threshold after %d rounds.\n", i);
            break;
        }

    }

    end = time(&end);
    params->iterations = i;
    printf("Returning after %.3f seconds.\n", difftime(end, start));

    return 0;

}
