#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"


#include <ctime>
#include <chrono>

void bgr2yuv(const Mat &input, Mat &output)
{
    vector<Mat> bgr;
    split(input, bgr);
    vector<Mat> yuv(3);
    for (size_t i = 0; i < yuv.size(); i++)
        yuv[i].create(bgr[i].size(), CV_32F);
    
    yuv[0] = 0.299   * bgr[2] + 0.587   * bgr[1] + 0.114   * bgr[0];
    yuv[1] = -0.14713* bgr[2] - 0.28886 * bgr[1] + 0.436   * bgr[0];
    yuv[2] = 0.615   * bgr[2] - 0.51499 * bgr[1] - 0.10001 * bgr[0];
    
    merge(yuv, output);
    
}

int main(int, char**)
{
        // Read the model .bin files  to .xml
        createUpperBodyCNNs();

        /* Load networks and modules */
        vector<string> files = {
                "../../../weights/20x16net.bin.xml"
                };

        cnn::CNN net20x16("20x16net");
        loadNet(files[0], net20x16);

        /* testing image for face detection */
        string imageFilename = "../../../test/img/38.png";
        Mat display = imread(imageFilename);
        Mat rgbimage = imread(imageFilename, IMREAD_COLOR),image, yuv, yuvN, resized;
        rgbimage.convertTo(image, CV_32F);
        image = image / 255.f;
    
    
        bgr2yuv(image, yuv);
    
        Scalar mean(0.47357687833093, -0.023079664067131, 0.022628687610046);
        Scalar stdev(0.24845277972345, 0.054840768814578, 0.076579628999438);
        Op::normMeanStd(yuv, yuvN, mean, stdev);

        double winSize = 16.;
        double minFaceSize = 30;
        double maxFaceSize = 180;
        double pyramidRate = sqrt(2.0);
        double faceSize = minFaceSize;
        double factor;

        cnn::CNNParam params;
        params.KernelH = 20;
        params.KernelW = 16;
        vector<Detection> outputs;
        vector<Detection> outputs20x16;
    
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
    

        while (faceSize < min(image.rows, image.cols) && faceSize < maxFaceSize)
        {
            
            factor = winSize/faceSize;
            resize(yuvN, resized, Size(0,0), factor, factor, INTER_AREA);

            
            cnn::faceDet::detect(resized, net20x16, params, outputs, .99f, 4.f);

            cnn::faceDet::backProject(outputs, factor);
            
            cnn::faceDet::displayResults(display, outputs, "20x16net");
            waitKey();
            
            outputs20x16.insert(outputs20x16.end(), outputs.begin(), outputs.end());
            faceSize *= pyramidRate;
            outputs.clear();
        }
        cnn::faceDet::displayResults(display, outputs20x16, "20x16net");

        end = std::chrono::system_clock::now();
        std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/ 1000.f) << " seconds" << std::endl;

        waitKey();

        return 0;
}
