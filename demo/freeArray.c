// free the second dimensional array
#include <stdlib.h>

void freeArray(float **img, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        free(img[i]);
    }
    free(img);
}
