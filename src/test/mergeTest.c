#include "global.h"
#include "merge.h"
#include "mergeSort.h"
#include <stdio.h>


int main(void){
    struct Windows* window;

    window[0].score = 2.56; window[0].x1 = 1;
    window[1].score = 7.25;window[1].x1 = 2;
    window[2].score = 3.45;window[2].x1 = 3;
    window[3].score = 5.5;window[3].x1 = 4;
    window[4].score = 1.45;window[4].x1 = 5;
    printf("%f\n", window[0].score);
    printf("%f\n", window[1].score);
    printf("%f\n", window[2].score);
    printf("%f\n", window[3].score);
    printf("%f\n", window[4].score);
    printf("----------------------\n");

    mergeSort(window, 0, 5);

    printf("%d, %f\n", window[0].x1, window[0].score);
    printf("%d, %f\n", window[1].x1, window[1].score);
    printf("%d, %f\n", window[2].x1, window[2].score);
    printf("%d, %f\n", window[3].x1, window[3].score);
    printf("%d, %f\n", window[4].x1, window[4].score);

    return 0;
}
