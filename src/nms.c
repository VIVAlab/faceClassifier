#include "global.h"

// intersection of union
float iou(struct Windows window1, struct Windows window2);

// NMS
void nms(struct Windows* window, int counter, float NMS_threshold){
    int i, j;
    for (i = 0; i < counter - 1; i++){
        if (window[i].dropped == true) continue;
        for (j = i + 1; j < counter; j++){
            // printf("#%d, score: %f, iou: %f,dropped: %s\n", j, window[j].score, window[j].iou, window[j].dropped ? "true" : "false");
            if (window[j].dropped == true) continue;
            if ( (window[j].iou = iou(window[i], window[j])) > NMS_threshold )
                // printf("iou: %f, threshold: %f\n", window[j].iou, NMS_threshold);
                window[j].dropped = true;
        }
    }
}
