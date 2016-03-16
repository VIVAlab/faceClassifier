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
        cnn::createCNNs();

        /* Load networks and modules */
        vector<string> files = {
                "../../../weights/12net.bin.xml",
        };

        cnn::CNN net("net");
        loadNet(files[0], net);

        /* testing image for face detection */
        string imageFilename = "../../../test/img/group1.jpg";
        Mat display = imread(imageFilename);
        Mat image = imread(imageFilename, IMREAD_GRAYSCALE), imageN, resized;

        image.convertTo(image, CV_32F);
        image = image / 255.f;

        cnn::Op::normGlobal(image, imageN);

        double winSize = 12.;
        double minFaceSize = 30;
        double maxFaceSize = 180;
        double pyramidRate = sqrt(2.0);
        double faceSize = minFaceSize;
        double factor;

        cnn::CNNParam params;
        params.KernelH = 12;
        params.KernelW = 12;
        vector<cnn::Detection> outputs;
        vector<cnn::Detection> outputs12;

        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
    
    
        while (faceSize < min(image.rows, image.cols) && faceSize < maxFaceSize)
        {
            factor = winSize/faceSize;
            resize(imageN, resized, Size(0,0), factor, factor, INTER_AREA);
            Mat score;
            cnn::Alg::detect(resized, net, params, outputs, score, .5f);

            Mat heatmap;
            cnn::Alg::heatMapFromScore(score, heatmap, image.size());
            imshow("heatmap", heatmap);
            waitKey();

            faceSize *= pyramidRate;
            outputs.clear();
        }


        end = std::chrono::system_clock::now();
        std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/ 1000.f) << " seconds" << std::endl;

        waitKey();

        return 0;
}
