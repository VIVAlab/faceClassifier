#include "global.h"

// segment overlap
int overlap(int a1, int a2, int b1, int b2){
    if (a2 <= b1 || b2 <= a1) return 0;
    if (a1 <= b1 && b2 <= a2) return b2 - b1;
    if (b1 <= a1 && a2 <= b2) return a2 - a1;
    if (a1 <= b1) return a2 - b1;
    return b2 - a1;
}

float iou(struct Windows window1, struct Windows window2){
    // overlap length in x and y dimendion
    int dx, dy;

    dx = overlap(window1.x1, window1.x2, window2.x1, window2.x2);
    if (dx == 0) return 0.0;

    dy = overlap(window1.y1, window1.y2, window2.y1, window2.y2);
    if (dy == 0) return 0.0;

    return (float)(dx * dy) / (float)((window1.x2 - window1.x1) * (window1.y2 - window1.y1) + (window2.x2 - window2.x1) * (window2.y2 - window2.y1) - dx * dy);
}
