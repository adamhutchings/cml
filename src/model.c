#include "model.h"

#include <assert.h>

int cmlmodelinit(struct cmlmodel * model, int insize, int outsize, int layers) {
    assert(model);
    cmlninit(&model->net, insize, outsize, layers);
    cmlnrandinit(&model->net);
    return 0;
}


int cmlmodelfree(struct cmlmodel * model) {
    assert(model);
    cmlnfree(&model->net);
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
