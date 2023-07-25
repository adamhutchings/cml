#include <assert.h>
#include <stdio.h>

#include "mvmath.h"

/**
 * Simple test: is this true:
 * -1  4  3   1   16
 *  3 -1  2 * 2 = 7
 *  0  1  8   3   26
 */
int mvtest() {
    struct cmlmatrix m;
    struct cmlvector v1, v2;
    cmlminit(&m, 3, 3);
    cmlvinit(&v1, 3);
    cmlvinit(&v2, 3);
    v1.entries[0] = 1;
    v1.entries[1] = 2;
    v1.entries[2] = 3;
    cmlmsentry(&m, 0, 0, -1);
    cmlmsentry(&m, 0, 1,  4);
    cmlmsentry(&m, 0, 2,  3);
    cmlmsentry(&m, 1, 0,  3);
    cmlmsentry(&m, 1, 1, -1);
    cmlmsentry(&m, 1, 2,  2);
    cmlmsentry(&m, 2, 0,  0);
    cmlmsentry(&m, 2, 1,  1);
    cmlmsentry(&m, 2, 2,  8);
    cmlmul(&m, &v1, &v2);
    assert(v2.entries[0] == 16);
    assert(v2.entries[1] == 7);
    assert(v2.entries[2] == 26);
    return 0;
}

int main(int argc, char ** argv) {
    printf("%s\n", "Hello, World!");
    mvtest();
    return 0;
}
