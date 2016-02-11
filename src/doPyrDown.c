// image pyramid down by rate
#include <cv.h>

IplImage* doPyrDown(IplImage *src, int rate){
    IplImage* result = cvCreateImage(cvSize(src -> width / rate, src -> height / rate), src -> depth, src -> nChannels);
    cvResize(src, result, CV_INTER_AREA);

    return result;
}
