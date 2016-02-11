#include "global.h"
#include <stdlib.h>

void merge(struct Windows* window, int lo, int mi, int hi){
    int i, j, k;
    // merged vector
    struct Windows* A = window + lo;

    // left sub-vector
    int lb = mi - lo; struct Windows* B = (struct Windows*)malloc(lb * sizeof(struct Windows));
    for (i = 0; i < lb; i++) B[i] = A[i];

    // right sub-vector
    int lc = hi - mi; struct Windows* C = window + mi;

    // the one with higher score from B[j] & C[k] is attached to the end of A
    for (i = 0, j = 0, k = 0; (j < lb) || (k < lc);){
        if ((j < lb) && (!(k < lc) || (B[j].score >= C[k].score))) A[i++] = B[j++];
        if ((k < lc) && (!(j < lb) || (C[k].score >= B[j].score))) A[i++] = C[k++];
    }

    free(B);
}

void mergeSort(struct Windows* window, int lo, int hi){
    if (hi - lo < 2) return;
    int mi = (lo + hi) / 2;
    mergeSort(window, lo, mi); mergeSort(window, mi, hi);
    merge(window, lo, mi, hi);
}
