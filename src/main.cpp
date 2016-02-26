#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"



void displayResults(Mat &image, vector<Detection> &detections, const string wName = "default")
{
    Mat tmp = image.clone();
    for (size_t i = 0; i < detections.size(); i++)
        rectangle(tmp, detections[i].face.tl(),
                         detections[i].face.br(), Scalar::all(255));
    imshow(wName, tmp);
    // waitKey();
}



int main(int, char**)
{
        // read .bin to .xml
        // binToXML();

        string imageFilename = "../../../test/img/group1.jpg";
        Mat image = imread(imageFilename, IMREAD_GRAYSCALE), resized;
        image.convertTo(image, CV_32F);
        image = image / 255.f;

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

        double winSize = 12.;
        double minFaceSize = 72;
        double factor = winSize/minFaceSize;
        resize(image, resized, Size(0,0), factor, factor, INTER_AREA);

        // 12 net
        vector<Detection> outputs;
        cnn::CNNParam params;
        params.StrideH = 4;
        params.StrideW = 4;
        params.KernelH = 12;
        params.KernelW = 12;

        cnn::faceDet::cascade(resized, params, net12, net12c, outputs, .5f, .1f);
    
        cnn::faceDet::backProjectDetections(outputs, factor);
        displayResults(image, outputs, "net12");
    
//        cnn::faceDet::nms(outputs, .2f);
    
        // displayResults(image, outputs, "net12 after nms");
    
        // 24 net
        cnn::faceDet::filterDetections(image, outputs, Size(24,24), net24, net24c, .5f, .1f);
//        cnn::faceDet::nms(outputs, .5f);

        displayResults(image, outputs, "net24");

        // 48 net
        cnn::faceDet::filterDetections(image, outputs, Size(48,48), net48, net48c, .5f, .1f);
//        cnn::faceDet::nms(outputs, .3f);

        displayResults(image, outputs, "net48");
        waitKey();

        return 0;
}
