// image pyramid down by rate
#include <cv.h>

IplImage* doPyrDown(IplImage *src, int rate){
    IplImage* result = cvCreateImage(cvSize(src -> width / rate, src -> height / rate), src -> depth, src -> nChannels);
    cvPyrDown(src, result, CV_GAUSSIAN_5x5);

    return result;
}
