#include "global.h"

// ***************** parameters settings start *************************
// home folder
const char FILE_PATH[] = "/Users/andres/Documents/Programming/github/faceClassifier/";

// test image path
const char TEST_IMAGE[] = "test/img/group1.jpg";

// minimum size(pixels) of detection object
const int MinImageSize = 16;

// number of test images(test on images in specific folder)
const int loopSize = 14266;

// demostration mode on(true)/ off(false)
const bool flagPrinting = true;

// ***************** parameters settings end ***************************

// function declarations
// layers
float Layer12(float **img, int width, int height, int channels);
float* CaliLayer12(float **img, int width, int height, int channels);
float Layer24(float **img, int width, int height, int channels);
float* CaliLayer24(float **img, int width, int height, int channels);
float Layer48(float **img, int width, int height, int channels);
float* CaliLayer48(float **img, int width, int height, int channels);

// convert numbers from int to char[]
char* itos(int i, char b[]);

// image pyramid down by rate
IplImage* doPyrDown(IplImage *src, int rate);

// preprocess image data
void preprocess(float **img, unsigned char *data, int row, int col, int step, int channels, int size);

// free two-dimensonal array with n rows
void freeArray(float **img, int n);

int main(void){
    IplImage *srcImg;
    IplImage *dstImg;
    int height, width, step, channels;
    uchar *data;
    int i, j, k;
    float mean_x;
    float mean_x2;
    float std = 0.0;
    float res;

    // initialize file path
    char path[50];
    char file[50];
    strcpy(path, FILE_PATH);
    strcat(path, "cnn/test/c_faces/pic");

    char *suffix = ".jpg";
    char imgNum[10];
    char *num;
    strcat(path, imgNum);

    int loop;
    int digits;

// image test loop starts
for (loop = 1; loop < loopSize; loop++){
    strcpy(file, path);
    digits = (int)log10f((float)loop);

    // add the zeros in the filename(for c_faces)
    while (digits < 5 - 1){
        num = itos(0, imgNum);
        strcat(file, num);
        digits++;
    }

    // add zeros in the filename(for nonfaces)
    /*
    while (digits < 4 - 1){
        num = itos(0, imgNum);
        strcat(file, num);
        digits++;
    }
    */

    num = itos(loop, imgNum);
    strcat(file, num);
    strcat(file, suffix);

    // ----------------------------------------------------------
    // for testing
    strcpy(file, FILE_PATH);
    strcat(file, TEST_IMAGE);
    // ----------------------------------------------------------

    printf("Testing on: %s\n", file);

    srcImg = cvLoadImage(file, CV_LOAD_IMAGE_GRAYSCALE);
    if (!srcImg){
        printf("Could not load image file: %s\n", file);
        continue;
    }

    // original size of the image
    IplImage *originalImg;
    originalImg = cvLoadImage(file, CV_LOAD_IMAGE_GRAYSCALE);
    int WIDTH = originalImg->width;
    int HEIGHT = originalImg->height;

    // detected objects image
    IplImage *detectedImg = cvCloneImage(originalImg);
    IplImage *detectedImg12 = cvCloneImage(originalImg);
    IplImage *detectedImg24 = cvCloneImage(originalImg);

    // object coordinate information
    int object[10000][4];
    int num_object = 0;

    // image pyramid loop starts
    while (srcImg -> height >= MinImageSize){

        // get the image data
        width = srcImg -> width;
        height = srcImg -> height;
        step = srcImg -> widthStep;
        channels = srcImg -> nChannels;
        data = (uchar*)srcImg -> imageData;

        const int Stride = 4;
        int row, col;
        float **img = malloc(12 * sizeof(float*));
        for (i = 0; i < 12; i++){
            img[i] = malloc(12 * sizeof(float));
        }

        // window sliding loop starts
        for (row = 0; row + 12 <= height; row += Stride){
            for (col = 0; col + 12 <= width; col += Stride){

                preprocess(img, data, row, col, step, channels, 12);

                // printf("%d%% tested, testing on image %s\n", (int)((float)loop*100/14266), file);

                // 12 net
                res = Layer12(img, 12, 12, channels);


                int static counter = 0;
                // 12 calibration
                if (res > 0.5){
                    counter++;

                    // printf("\n\n---------- face detected at row: %d, col: %d ------------\n\n", row, col);

                    float *out_12c;
                    out_12c = CaliLayer12(img, 12, 12, channels);

                    // printf("detection: %f, %f, %f\n", out_12c[0], out_12c[1], out_12c[2]);
                    float s, x, y;
                    s = out_12c[0];
                    x = out_12c[1];
                    y = out_12c[2];

                    free(out_12c);

                    // NAN values have odd property(comparison involving them are always false)
                    if (s != s || x != x || y != y)
                        continue;

                    // real position and size of the detected window in original image
                    int multiplicant;
                    int realPos_w, realPos_h , realWinSize_w, realWinSize_h;

                    multiplicant = WIDTH / width;

                    realWinSize_w = 12 * multiplicant;
                    realWinSize_h = 12 * multiplicant;
                    realPos_w = col * multiplicant;
                    realPos_h = row * multiplicant;

                    // calibration
                    int cali_x, cali_y, cali_w, cali_h;

                    cali_x = (int)(realPos_w - x * realWinSize_w / s);
                    cali_y = (int)(realPos_h - y * realWinSize_h / s);
                    cali_w = (int)(realWinSize_w / s);
                    cali_h = (int)(realWinSize_h / s);

                    // make sure the calibration image not beyond the boundary
                    if (cali_x >= WIDTH || cali_y >= HEIGHT){
                        continue;
                    }
                    cali_x = max(cali_x, 0);
                    cali_y = max(cali_y, 0);
                    cali_w = min(cali_w, WIDTH - cali_x);
                    cali_h = min(cali_h, HEIGHT - cali_y);

                    // ----------- for testing -----------------
                    // skip the small windows
                    if (width > 200)
                        continue;
                    // -----------------------------------------

                    /*
                    printf("real window size: %d * %d,  position row: %d, col: %d\n", realWinSize_w, realWinSize_h, realPos_w, realPos_h);
                    printf("x: %f, y: %f, s: %f\n", x, y, s);
                    printf("cali window position: %d, %d,  cali position width: %d, height: %d\n", cali_x, cali_y, cali_w, cali_h);
                    */


                    // show the input of 12 net
                    if (flagPrinting){
                    cvSetImageROI(originalImg, cvRect(realPos_w, realPos_h, realWinSize_w, realWinSize_h));
                    IplImage *input12Img= cvCreateImage(cvSize(12, 12), IPL_DEPTH_8U, 1);
                    cvResize(originalImg, input12Img, CV_INTER_AREA);

                    IplImage *temp12Img = cvCreateImage(cvSize(100, 100), IPL_DEPTH_8U, 1);
                    cvResize(input12Img, temp12Img, CV_INTER_AREA);
                    cvNamedWindow("12 net", CV_WINDOW_AUTOSIZE);
                    cvMoveWindow("12 net", 1150, 10);
                    cvShowImage("12 net", temp12Img);

                    cvResetImageROI(originalImg);
                    }

                    if (flagPrinting){
                        IplImage *origImg_cali = cvCloneImage(originalImg);
                        cvNamedWindow("12 calibration", CV_WINDOW_AUTOSIZE);
                        cvMoveWindow("12 calibration", 350, 20);
                        cvRectangle(origImg_cali, cvPoint(cali_x, cali_y), cvPoint(cali_x + cali_w, cali_y + cali_h), cvScalar(255, 0, 0, 0), 2, 4, 0);
                        cvRectangle(detectedImg12, cvPoint(cali_x, cali_y), cvPoint(cali_x + cali_w, cali_y + cali_h), cvScalar(255, 0, 0, 0), 2, 4, 0);
                        cvShowImage("12 calibration", origImg_cali);
                    }


                    float **img24 = malloc(24 * sizeof(float*));
                    for (i = 0; i < 24; i++){
                        img24[i] = malloc(24 * sizeof(float));
                    }

                    // 24 net
                    // resize the detected window of original image after 12 calibration net
                    cvSetImageROI(originalImg, cvRect(cali_x, cali_y, cali_w, cali_h));
                    IplImage *input24Img = cvCreateImage(cvSize(24, 24), IPL_DEPTH_8U, 1);
                    cvResize(originalImg, input24Img, CV_INTER_AREA);

                    if (flagPrinting){
                        IplImage *tempImg = cvCreateImage(cvSize(100, 100), IPL_DEPTH_8U, 1);
                        cvResize(input24Img, tempImg, CV_INTER_AREA);
                        cvNamedWindow("24 net", CV_WINDOW_AUTOSIZE);
                        cvMoveWindow("24 net", 1150, 165);
                        cvShowImage("24 net", tempImg);
                    }
                    uchar *data24;
                    data24 = (uchar*) input24Img -> imageData;

                    preprocess(img24, data24, 0, 0, input24Img->widthStep, input24Img->nChannels, 24);

                    res = Layer24(img24, 24, 24, input24Img->nChannels);

                    if (res > 0.5){

                        // 24 calibration
                        float *out_24c;
                        out_24c = CaliLayer24(img24, 24, 24, input24Img->nChannels);
                        float s_24, x_24, y_24;
                        s_24 = out_24c[0];
                        x_24 = out_24c[1];
                        y_24 = out_24c[2];
                        free(out_24c);
                        // printf("s: %f, x: %f, y: %f\n", s_24, x_24, y_24);

                        int cali24_x, cali24_y, cali24_w, cali24_h;

                        cali24_x = (int)(cali_x - x_24 * cali_w / s_24);
                        cali24_y = (int)(cali_y - y_24 * cali_h / s_24);
                        cali24_w = (int)(cali_w / s_24);
                        cali24_h = (int)(cali_h / s_24);

                        // printf("24_x: %d, 24_y: %d, 24_w: %d, 24_h: %d\n", cali24_x, cali24_y, cali24_w, cali24_h);

                        if (flagPrinting){
                        cvResetImageROI(originalImg);

                        IplImage *origImg_cali24 = cvCloneImage(originalImg);
                        cvNamedWindow("24 calibration", CV_WINDOW_AUTOSIZE);
                        cvMoveWindow("24 calibration", 0, 350);
                        cvRectangle(origImg_cali24, cvPoint(cali24_x, cali24_y), cvPoint(cali24_x + cali24_w, cali24_y + cali24_h), cvScalar(255, 0, 0, 0), 2, 4, 0);
                        cvRectangle(detectedImg24, cvPoint(cali24_x, cali24_y), cvPoint(cali24_x + cali24_w, cali24_y + cali24_h), cvScalar(255, 0, 0, 0), 2, 4, 0);
                        cvShowImage("24 calibration", origImg_cali24);
                        }



                        // 48 net
                        float **img48= malloc(48 * sizeof(float*));
                        for (i = 0; i < 48; i++){
                            img48[i] = malloc(48 * sizeof(float));
                        }

                        // resize the detected window of original image after 12 calibration net
                        cvSetImageROI(originalImg, cvRect(cali24_x, cali24_y, cali24_w, cali24_h));
                        IplImage *input48Img = cvCreateImage(cvSize(48, 48), IPL_DEPTH_8U, 1);
                        cvResize(originalImg, input48Img, CV_INTER_AREA);;

                    if (flagPrinting){
                        IplImage *temp48Img = cvCreateImage(cvSize(100, 100), IPL_DEPTH_8U, 1);
                        cvResize(input48Img, temp48Img, CV_INTER_AREA);;
                        cvNamedWindow("48 net", CV_WINDOW_AUTOSIZE);
                        cvMoveWindow("48 net", 1150, 300);
                        cvShowImage("48 net", temp48Img);

                        cvWaitKey(100);
                    }

                        uchar *data48;
                        data48 = (uchar*) input48Img -> imageData;

                        preprocess(img48, data48, 0, 0, input48Img->widthStep, input48Img->nChannels, 48);

                        res = Layer48(img48, 48, 48, input48Img->nChannels);
                        if (res > 0.5){
                            printf("\n48 layer: PASS\n");



                        // 48 calibration
                        float *out_48c;
                        out_48c = CaliLayer48(img48, 48, 48, input48Img->nChannels);

                        float s_48, x_48, y_48;
                        s_48 = out_48c[0];
                        x_48 = out_48c[1];
                        y_48 = out_48c[2];

                        free(out_48c);

                        /*
                        // NAN values have odd property(comparison involving them are always false)
                        if (s_48 != s_48 || x_48 != x_48 || y_48 != y_48)
                            continue;
                        */
                        printf("s: %f, x: %f, y: %f\n", s_48, x_48, y_48);

                        int cali48_x, cali48_y, cali48_w, cali48_h;

                        cali48_x = (int)(cali_x - x_48 * cali_w / s_48);
                        cali48_y = (int)(cali_y - y_48 * cali_h / s_48);
                        cali48_w = (int)(cali_w / s_48);
                        cali48_h = (int)(cali_h / s_48);

                        cvResetImageROI(originalImg);

                        if (flagPrinting){
                        IplImage *origImg_cali48 = cvCloneImage(originalImg);
                        cvNamedWindow("48 calibration", CV_WINDOW_AUTOSIZE);
                        cvMoveWindow("48 calibration", 600, 350);
                        cvRectangle(origImg_cali48, cvPoint(cali24_x, cali24_y), cvPoint(cali24_x + cali24_w, cali24_y + cali24_h), cvScalar(255, 0, 0, 0), 2, 4, 0);
                        cvShowImage("48 calibration", origImg_cali48);

                        cvWaitKey(300);
                        cvDestroyWindow("48 calibration");
                        }

                        // 48 calibration
                        // cvRectangle(origImg_cali48, cvPoint(cali48_x, cali48_y), cvPoint(cali48_x + cali48_w, cali48_y + cali48_h), cvScalar(255, 0, 0, 0), 2, 4, 0);

                        // detection result
                        cvRectangle(detectedImg, cvPoint(cali24_x, cali24_y), cvPoint(cali24_x + cali24_w, cali24_y + cali24_h), cvScalar(255, 0, 0, 0), 2, 4, 0);














                        } else {
                            printf("\n48 layer: fail\n");
                        }

                        freeArray(img48, 48);
                    }

                    if (flagPrinting){
                    cvWaitKey(100);

                    cvDestroyWindow("48 net");
                    cvDestroyWindow("24 calibration");

                    // reset to original image
                    cvResetImageROI(originalImg);
                    }

                    freeArray(img24, 24);
                }

            }
        }
        // window sliding loop ends

        freeArray(img, 12);

        // down pyramid
        dstImg = doPyrDown(srcImg, 2);
        srcImg = dstImg;

        } // image pyramid loop end

        cvDestroyWindow("12 calibration");
        cvDestroyWindow("12 net");
        cvDestroyWindow("24 net");


        cvNamedWindow("12 result", CV_WINDOW_AUTOSIZE);
        cvMoveWindow("12 result", 100, 50);
        cvShowImage("12 result", detectedImg12);

        cvNamedWindow("24 result", CV_WINDOW_AUTOSIZE);
        cvMoveWindow("24 result", 300, 50);
        cvShowImage("24 result", detectedImg24);


        cvNamedWindow("detection result", CV_WINDOW_AUTOSIZE);
        cvMoveWindow("detection result", 200, 350);
        cvShowImage("detection result", detectedImg);
        cvWaitKey(0);
        cvDestroyWindow("12 result");
        cvDestroyWindow("24 result");
        cvDestroyWindow("detection result");

        exit(0);
    }
    // image test loop ends

    return 0;
}
