#include "global.h"

// ***************** parameters settings start *************************
// path of cnn folder
const char FILE_PATH[] = "/home/binghao/faceClassifier/";

// test image path
const char TEST_IMAGE[] = "test/img/group1.jpg";

// minimum size(pixels) of detection object (multiple of 12)
const int MinFaceSize = 72;

// thresholds
const float Threshold_12Layer = 0.5;
const float Threshold_24Layer = 0.5;
const float Threshold_48Layer = 0.5;
const float Threshold_12NMS = 0.3;
const float Threshold_24NMS = 0.3;
const float Threshold_48NMS = 0.3;

// ***************** parameters settings end ***************************

// function declarations
// layers
float Layer12(float **img, int width, int height, int channels);
float* CaliLayer12(float **img, int width, int height, int channels);
float Layer24(float **img, int width, int height, int channels);
float* CaliLayer24(float **img, int width, int height, int channels);
float Layer48(float **img, int width, int height, int channels);
float* CaliLayer48(float **img, int width, int height, int channels);

// image pyramid down by rate
IplImage* doPyrDown(IplImage *src, int rate);

// preprocess image data
void preprocess(float **img, unsigned char *data, int row, int col, int step, int channels, int size);

// sort function
void mergeSort(struct Windows* window, int lo, int hi);

// free two-dimensonal array with n rows
void freeArray(float **img, int n);

// NMS
void nms(struct Windows window[], int counter, float NMS_threshold);

int main(void){
    // detection windows
    struct Windows window[500];

    // loop counter
    int i, j, k;
    int row, col;
    int counter = 0;    // detection window counter

    // image information
    int height, width, step, channels;
    uchar *data, *data24, *data48;

    // size, x, y for calibration
    float *out_12c, *out_24c, *out_48c;     // vector carrying s,x,y
    float s, x, y;
    int cali_x, cali_y, cali_w, cali_h;

    // scores of the 12 layer
    float res_12Layer;
    float res_24Layer;
    float res_48Layer;

    // window sliding stride
    const int Stride = 4;

    // image pyramid rate
    int pyr_rate = MinFaceSize / 12;

    // image pyrimid stopping
    bool flagStop = false;

    // file path
    char file[50];
    strcpy(file, FILE_PATH);
    strcat(file, TEST_IMAGE);

    // alloc memory for 12x12 image
    float **img = malloc(12 * sizeof(float*));
    for (i = 0; i < 12; i++){
        img[i] = malloc(12 * sizeof(float));
    }

    // alloc memory for 24x24 image
    float **img24 = malloc(24 * sizeof(float*));
    for (i = 0; i < 24; i++){
        img24[i] = malloc(24 * sizeof(float));
    }

    // alloc memory for 48x48 image
    float **img48 = malloc(48 * sizeof(float*));
    for (i = 0; i < 48; i++){
        img48[i] = malloc(48 * sizeof(float));
    }

    // for printing scores
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.3, 0.3, 0, 1, 8);
    char word[5];

    // load image
    IplImage *srcImg, *dstImg;
    srcImg = cvLoadImage(file, CV_LOAD_IMAGE_GRAYSCALE);

    // original size of the image
    const int WIDTH = srcImg->width;
    const int HEIGHT = srcImg->height;

    IplImage *originImg = cvCloneImage(srcImg);
    IplImage *originImg1 = cvCloneImage(srcImg);
    IplImage *originImg2 = cvCloneImage(srcImg);
    IplImage *originImg3 = cvCloneImage(srcImg);
    IplImage *input24Img = cvCreateImage(cvSize(24, 24), IPL_DEPTH_8U, 1);
    IplImage *input48Img = cvCreateImage(cvSize(48, 48), IPL_DEPTH_8U, 1);

    if (!srcImg){
        printf("Could not load image file: %s\n", file);
        exit(1);
    }


    // image pyramid loop starts
    while (!flagStop){
        counter = 0;

        // image pyramid down
        dstImg = doPyrDown(srcImg, pyr_rate);

        // get the image data
        width = dstImg -> width;
        height = dstImg -> height;
        step = dstImg -> widthStep;
        channels = dstImg -> nChannels;
        data = (uchar*)dstImg -> imageData;

        IplImage *detectedImg_12Layer = cvCloneImage(dstImg);

        // window sliding loop starts
        for (row = 0; row + 12 <= height; row += Stride){
            for (col = 0; col + 12 <= width; col += Stride){
                // 12 layer, 12 calibration, NMS
                preprocess(img, data, row, col, step, channels, 12);

                res_12Layer = Layer12(img, 12, 12, channels);

                // 12 layer passed
                if (res_12Layer > Threshold_12Layer){
                    // 12 calibration layer
                    out_12c = CaliLayer12(img, 12, 12, channels);

                    s = out_12c[0]; x = out_12c[1]; y = out_12c[2];
                    free(out_12c);      // memory allocated in CaliLayer12

                    // ignoring returned NAN values(comparison involving them are always false)
                    if (s != s || x != x || y != y) continue;

                    // calibration
                    cali_x = col * pyr_rate - x * 12 * pyr_rate / s;
                    cali_y = row * pyr_rate - y * 12 * pyr_rate / s;
                    cali_w = 12 * pyr_rate / s;
                    cali_h = 12 * pyr_rate / s;

                    // make sure the calibrated window not beyond the image boundary
                    if (cali_x >= WIDTH || cali_y >= HEIGHT) continue;

                    cali_x = max(cali_x, 0);
                    cali_y = max(cali_y, 0);
                    cali_w = min(cali_w, WIDTH - cali_x);
                    cali_h = min(cali_h, HEIGHT - cali_y);

                    window[counter].x1 = cali_x;                    // x1
                    window[counter].y1 = cali_y;                    // y1
                    window[counter].x2 = cali_x + cali_w;           // x2
                    window[counter].y2 = cali_y + cali_h;           // y2
                    window[counter].score = res_12Layer;            // 12 layer score
                    window[counter].iou = 0.0;                      // iou ratio
                    window[counter].dropped= false;                 // if it's dropped
                    counter++;
                    // end of 12 layer, 12 calibration


                }
            }
        }
        // window sliding loop ends

        // sort the detection windows by score in descending order
        mergeSort(window, 0, counter);

        // display sorted windows surviving 12 layer
        cvNamedWindow("12 layer", CV_WINDOW_AUTOSIZE);
        for (i = 0; i < counter; i++){
            cvRectangle(originImg, cvPoint(window[i].x1, window[i].y1), cvPoint(window[i].x2, window[i].y2), cvScalar(255,0,0,0), 2, 4, 0);
            // printf("[#%d] x1: %d, y1: %d, x2: %d, y2: %d, score: %f, iou: %f, dropped: %s\n", i, window[i].x1, window[i].y1, window[i].x2, window[i].y2, window[i].score, window[i].iou, window[i].dropped ? "true" : "false");

            if (window[i].dropped == false){
                sprintf(word, "%.2f", window[i].score);
                cvPutText(originImg, word, cvPoint(window[i].x1, window[i].y1), &font, cvScalar(255, 255, 255, 0));
            }
        }
        cvShowImage("12 layer", originImg);
        cvMoveWindow("12 layer", 10, 10);

        printf("12 layer: x1: %d, y1: %d, x2: %d, y2: %d\n", window[15].x1, window[15].y1, window[15].x2, window[15].y2);

        // NMS after 12 calibration
        nms(window, counter, Threshold_12NMS);

        // display sorted windows surviving 12 layer
        cvNamedWindow("12 layer after NMS", CV_WINDOW_AUTOSIZE);
        for (i = 0; i < counter; i++){
            if (window[i].dropped == false){
                cvRectangle(originImg1, cvPoint(window[i].x1, window[i].y1), cvPoint(window[i].x2, window[i].y2), cvScalar(255,0,0,0), 2, 4, 0);

                sprintf(word, "%.2f", window[i].score);
                cvPutText(originImg1, word, cvPoint(window[i].x1, window[i].y1), &font, cvScalar(255, 255, 255, 0));
            }
        }
        cvShowImage("12 layer after NMS", originImg1);
        cvMoveWindow("12 layer after NMS", 500, 10);

        // 24 layer, 24 calibration, NMS
        for (i = 0; i< counter; i++){
            if (window[i].dropped == true) continue;

            cvSetImageROI(srcImg, cvRect(window[i].x1, window[i].y1, window[i].x2 - window[i].x1, window[i].y2 - window[i].y1));
            cvResize(srcImg, input24Img, CV_INTER_AREA);

            data24 = (uchar*) input24Img->imageData;

            preprocess(img24, data24, 0, 0, input24Img->widthStep, input24Img->nChannels, 24);
            res_24Layer = Layer24(img24, 24, 24, input24Img->nChannels);

            // 24 layer passed
            if (res_24Layer > Threshold_24Layer){
                // 24 calibration
                out_24c = CaliLayer24(img24, 24, 24, input24Img->nChannels);
                s = out_24c[0];
                x = out_24c[1];
                y = out_24c[2];
                free(out_24c);

                if (i == 15)
                    printf("s: %f, x: %f, y: %f\n", s, x, y);

                cali_x = window[i].x1 - x * (window[i].x2 - window[i].x1) / s;
                cali_y = window[i].y1 - y * (window[i].y2 - window[i].y1) / s;
                cali_w = (window[i].x2 - window[i].x1) / s;
                cali_h = (window[i].y2 - window[i].y1) / s;


                if (i == 15)
                    printf("cali_x: %d, cali_y: %d, cali_w: %d, cali_h: %d\n", cali_x, cali_y, cali_w, cali_h);

                // make sure the calibrated window not beyond the image boundary
                if (cali_x >= WIDTH || cali_y >= HEIGHT) continue;

                cali_x = max(cali_x, 0);
                cali_y = max(cali_y, 0);
                cali_w = min(cali_w, WIDTH - cali_x);
                cali_h = min(cali_h, HEIGHT - cali_y);

                window[i].x1 = cali_x;                    // x1
                window[i].y1 = cali_y;                    // y1
                window[i].x2 = cali_x + cali_w;           // x2
                window[i].y2 = cali_y + cali_h;           // y2
                window[i].score = res_24Layer;            // 12 layer score
                window[i].iou = 0.0;                      // iou ratio
                window[i].dropped= false;                 // if it's dropped

            }

            cvResetImageROI(srcImg);
        }

        printf("24 layer: x1: %d, y1: %d, x2: %d, y2: %d\n", window[15].x1, window[15].y1, window[15].x2, window[15].y2);

        // NMS after 24 calibration
        nms(window, counter, Threshold_24NMS);

        // display sorted windows surviving 24 layer
        cvNamedWindow("24 layer", CV_WINDOW_AUTOSIZE);
        for (i = 0; i < counter; i++){
            if (window[i].dropped == false){
                cvRectangle(originImg2, cvPoint(window[i].x1, window[i].y1), cvPoint(window[i].x2, window[i].y2), cvScalar(255,0,0,0), 2, 4, 0);

                sprintf(word, "%.2f", window[i].score);
                cvPutText(originImg2, word, cvPoint(window[i].x1, window[i].y1), &font, cvScalar(255, 255, 255, 0));
            }
        }
        cvShowImage("24 layer", originImg2);
        cvMoveWindow("24 layer", 10, 400);
        // end of 24 layer, 24 calibration, NMS


        // 48 layer, 48 calibration, NMS
        for (i = 0; i< counter; i++){
            if (window[i].dropped == true) continue;

            cvSetImageROI(srcImg, cvRect(window[i].x1, window[i].y1, window[i].x2 - window[i].x1, window[i].y2 - window[i].y1));
            cvResize(srcImg, input48Img, CV_INTER_AREA);

            data48 = (uchar*) input48Img->imageData;

            preprocess(img48, data48, 0, 0, input48Img->widthStep, input48Img->nChannels, 48);
            res_48Layer = Layer48(img48, 48, 48, input48Img->nChannels);

            // 48 layer passed
            if (res_48Layer > Threshold_48Layer){
                // 48 calibration
                out_48c = CaliLayer48(img48, 48, 48, input48Img->nChannels);
                s = out_48c[0];
                x = out_48c[1];
                y = out_48c[2];
                free(out_48c);

                cali_x = window[i].x1 - x * (window[i].x2 - window[i].x1) / s;
                cali_y = window[i].y1 - y * (window[i].y2 - window[i].y1) / s;
                cali_w = (window[i].x2 - window[i].x1) / s;
                cali_h = (window[i].y2 - window[i].y1) / s;

                // make sure the calibrated window not beyond the image boundary
                if (cali_x >= WIDTH || cali_y >= HEIGHT) continue;

                cali_x = max(cali_x, 0);
                cali_y = max(cali_y, 0);
                cali_w = min(cali_w, WIDTH - cali_x);
                cali_h = min(cali_h, HEIGHT - cali_y);

                window[i].x1 = cali_x;                    // x1
                window[i].y1 = cali_y;                    // y1
                window[i].x2 = cali_x + cali_w;           // x2
                window[i].y2 = cali_y + cali_h;           // y2
                window[i].score = res_48Layer;            // 12 layer score
                window[i].iou = 0.0;                      // iou ratio
                window[i].dropped= false;                 // if it's dropped
            }

            cvResetImageROI(srcImg);
        }

        // NMS after 48 calibration
        nms(window, counter, Threshold_48NMS);

        // display sorted windows surviving 48 layer
        cvNamedWindow("48 layer", CV_WINDOW_AUTOSIZE);
        for (i = 0; i < counter; i++){
            if (window[i].dropped == false){
                cvRectangle(originImg3, cvPoint(window[i].x1, window[i].y1), cvPoint(window[i].x2, window[i].y2), cvScalar(255,0,0,0), 2, 4, 0);

                sprintf(word, "%.2f", window[i].score);
                cvPutText(originImg3, word, cvPoint(window[i].x1, window[i].y1), &font, cvScalar(255, 255, 255, 0));
            }
        }
        cvShowImage("48 layer", originImg3);
        cvMoveWindow("48 layer", 500, 400);
        // end of 48 layer, 48 calibration, NMS

        printf("48 layer: x1: %d, y1: %d, x2: %d, y2: %d, dropped: %s\n", window[15].x1, window[15].y1, window[15].x2, window[15].y2, window[15].dropped?"true":"false");







        cvWaitKey(0);
        cvDestroyWindow("12 layer");
        cvDestroyWindow("12 layer after NMS");
        cvDestroyWindow("24 layer");
        cvDestroyWindow("48 layer");

        pyr_rate *= 2;
        if (dstImg->height / 2 < 12) flagStop = true;
    }
    // image pyramid loop ends

    freeArray(img, 12);
    freeArray(img24, 12);
    freeArray(img48, 12);

    return 0;
}
