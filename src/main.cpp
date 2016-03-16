#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"


#include <ctime>
#include <chrono>


int main(int, char**)
{
        // Read the model .bin files  to .xml

        cnn::createUpperBodyCNNs();


        /* Load networks and modules */
        vector<string> files = {
                "../../../weights/20x16net.bin.xml"
                };

        cnn::CNN net20x16("20x16net");
        cnn::loadNet(files[0], net20x16);

        /* testing image for face detection */
        string imageFilename = "../../../test/img/38.png";
        Mat display = imread(imageFilename);
        Mat rgbimage = imread(imageFilename, IMREAD_COLOR),image, yuv, yuvN, resized;
        rgbimage.convertTo(image, CV_32F);
        image = image / 255.f;
    
    
        cnn::Op::bgr2yuv(image, yuv);
    
        Scalar mean(0.47357687833093, -0.023079664067131, 0.022628687610046);
        Scalar stdev(0.24845277972345, 0.054840768814578, 0.076579628999438);
        cnn::Op::normMeanStd(yuv, yuvN, mean, stdev);

        double winSize = 16.;
        double minFaceSize = 30;
        double maxFaceSize = 180;
        double pyramidRate = sqrt(2.0);
        double faceSize = minFaceSize;
        double factor;

        cnn::CNNParam params;

        params.KernelH = 20;
        params.KernelW = 16;
        vector<cnn::Detection> outputs;
        vector<cnn::Detection> outputs20x16;

        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
    

        while (faceSize < min(image.rows, image.cols) && faceSize < maxFaceSize)
        {
            
            factor = winSize/faceSize;
            resize(yuvN, resized, Size(0,0), factor, factor, INTER_AREA);

            Mat score;
            cnn::Alg::detect(resized, net20x16, params, outputs, score, .99f, 4.f);

            cnn::Alg::backProject(outputs, factor);
            
            cnn::Alg::displayResults(display, outputs, "20x16net");
            waitKey();
            
            outputs20x16.insert(outputs20x16.end(), outputs.begin(), outputs.end());
            faceSize *= pyramidRate;
            outputs.clear();
        }
        cnn::Alg::displayResults(display, outputs20x16, "20x16net");

        end = std::chrono::system_clock::now();
        std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/ 1000.f) << " seconds" << std::endl;

        waitKey();

        return 0;
}
