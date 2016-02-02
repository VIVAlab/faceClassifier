#include "global.h"

// ***************** parameters settings start *************************
// home folder
const char FILE_PATH[] = "/home/binghao/";

// test image path
const char TEST_IMAGE[] = "cnn/test/img/group1.jpg";

// minimum size(pixels) of detection object
const int MinImageSize = 16;

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

// free two-dimensonal array with n rows
void freeArray(float **img, int n);

int main(void){
    // loop counter
    int i, j, k;
    int row, col;

    // image information
    int height, width, step, channels;
    uchar *data;

    // scores of the 12 layer
    float res_12Layer;

    // window sliding stride
    const int Stride = 4;

    // file path
    char file[50];
    strcpy(file, FILE_PATH);
    strcat(file, TEST_IMAGE);

    // alloc memory for 12x12 image
    float **img = malloc(12 * sizeof(float*));
    for (i = 0; i < 12; i++){
        img[i] = malloc(12 * sizeof(float));
    }

    printf("Testing on: %s\n", file);

    // load image
    IplImage *srcImg;
    srcImg = cvLoadImage(file, CV_LOAD_IMAGE_GRAYSCALE);
    if (!srcImg){
        printf("Could not load image file: %s\n", file);
        exit(1);
    }

    // image pyramid loop starts
    while (srcImg -> height >= MinImageSize){

        // get the image data
        width = srcImg -> width;
        height = srcImg -> height;
        step = srcImg -> widthStep;
        channels = srcImg -> nChannels;
        data = (uchar*)srcImg -> imageData;

        // window sliding loop starts
        for (row = 0; row + 12 <= height; row += Stride){
            for (col = 0; col + 12 <= width; col += Stride){
                preprocess(img, data, row, col, step, channels, 12);

                // 12 layer start
                res_12Layer = Layer12(img, 12, 12, channels);

                if (res_12Layer > 0.5){
                    int static counter = 0;
                    printf("%f, #%d\n", res_12Layer, counter);
                    counter++;
                }
                // 12 layer end
            }
        }
        // window sliding loop ends
        exit(0);
    }
    // image pyramid loop ends

    freeArray(img, 12);

    return 0;
}
