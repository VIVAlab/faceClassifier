#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"


int main(int, char**)
{
        // read .bin to .xml
        //binToXML();

        /* Load networks and modules */
        vector<string> files = {
                "../../../weights/12net.bin.xml",
                "../../../weights/12cnet.bin.xml",
                "../../../weights/24net.bin.xml",
                "../../../weights/24cnet.bin.xml",
                "../../../weights/48net.bin.xml",
                "../../../weights/48cnet.bin.xml",
                };

        cnn::CNN net12("12net");
        cnn::CNN net12c("12cnet");
        cnn::CNN net24("24net");
        cnn::CNN net24c("24cnet");
        cnn::CNN net48("48net");
        cnn::CNN net48c("48cnet");
        loadNet(files[0], net12);
        loadNet(files[1], net12c);
        loadNet(files[2], net24);
        loadNet(files[3], net24c);
        loadNet(files[4], net48);
        loadNet(files[5], net48c);

        /* testing image for face detection */
        string imageFilename = "../../../test/img/group1.jpg";

        Mat image = imread(imageFilename, IMREAD_GRAYSCALE), display, imageN, resized;
        image.convertTo(image, CV_32F);
        image = image / 255.f;
        display = image.clone();

        Op::normGlobal(image, imageN);

        double winSize = 12.;
        double minFaceSize = 30;
        double maxFaceSize = 150;
        double pyramidRate = 1.3;
        double faceSize = minFaceSize;
        double factor;

        cnn::CNNParam params;
        params.KernelH = 12;
        params.KernelW = 12;
        vector<Detection> g_outputs;

        while (faceSize < min(image.rows, image.cols) && faceSize < maxFaceSize)
        {
            factor = winSize/faceSize;
            resize(imageN, resized, Size(0,0), factor, factor, INTER_AREA);
            
            vector<Detection> outputs;
            cnn::faceDet::detect(resized, net12, params, outputs, .5f);
            cnn::faceDet::nms(outputs, .1f);
            cnn::faceDet::calibrate(resized, net12c, outputs, 0.1f);
            cnn::faceDet::nms(outputs, .1f);
            cnn::faceDet::backProject(outputs, factor);
            cnn::faceDet::displayResults(display, outputs, "nms net12");
            
            waitKey();
            g_outputs.insert(g_outputs.end(), outputs.begin(), outputs.end());
            faceSize *= pyramidRate;
        }
        cnn::faceDet::nms(g_outputs, .1f);
        cnn::faceDet::displayResults(display, g_outputs, "results");

        waitKey();

        return 0;
}
