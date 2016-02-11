#include "global.h"

void mergeSort(struct Windows* window, int lo, int hi){
    if (hi - lo < 2) return;
    int mi = (lo + hi) / 2;
    mergeSort(window, lo, mi); mergeSort(window, mi, hi);
    merge(window, lo, mi, hi);
}
