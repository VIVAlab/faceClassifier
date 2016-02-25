#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"

void backProjectDetections(vector<Detection> &detects, const double &factor)
{
    for (size_t i = 0 ; i < detects.size(); i++)
    {
        detects[i].face.x /= factor;
        detects[i].face.y /= factor;
        detects[i].face.width  /= factor;
        detects[i].face.height /= factor;
    }
}

void filterDetections(Mat &image, vector<Detection> &detects, Size size,
                      cnn::CNN &cnn, cnn::CNN &cnet,
                      float thr = 0.5f ,
                      float calibThr = 0.1f)
{
    for (size_t i = 0; i < detects.size(); i++)
    {
        Mat _resizedROI, _normROI, _netOutput;
        Rect imageROI(0,0, image.cols, image.rows);
        resize(image(detects[i].face & imageROI), _resizedROI, size);
        Op::norm(_resizedROI, _normROI);
        cnn.forward(_normROI, _netOutput);
        
        if (_netOutput.at<float>(0) > thr)
        {
            detects[i].score = _netOutput.at<float>(0);
            
            Mat _cnetOutput;
            cnet.forward(_normROI, _cnetOutput);
            detects[i] = cnn::Op::applyTransformationCode(detects[i], _cnetOutput, .1f);
        }
        else
        {
            detects.erase(detects.begin() + i);
            i--;
        }
    }
}

void displayResults(Mat &image, vector<Detection> &detections, const string wName = "default")
{
    Mat tmp = image.clone();
    for (size_t i = 0; i < detections.size(); i++)
        rectangle(tmp, detections[i].face.tl(),
                         detections[i].face.br(), Scalar::all(255));
    imshow(wName, tmp);
    waitKey();
}

void loadNet(const string &filename, cnn::CNN &net)
{
    FileStorage fs(filename, FileStorage::READ);
    fs["cnn"] >> net;
    fs.release();
}

int main(int, char**)
{
    
        string imageFilename = "../../..//test/img/group1.jpg";
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
        cnn::CNN net48("48net", true);
        cnn::CNN net48c("48cnet");
        loadNet(files[0], net12);
        loadNet(files[1], net12c);
        loadNet(files[2], net24);
        loadNet(files[3], net24c);
        loadNet(files[4], net48);
        loadNet(files[5], net48c);

        double winSize = 12.;
        double minFaceSize = 60.;
        double factor = winSize/minFaceSize;
        resize(image, resized, Size(0,0), factor, factor, INTER_AREA);

        // 12 net
        vector<Detection> outputs;
        cnn::CNNParam params;
        params.StrideH = 4;
        params.StrideW = 4;
        params.KernelH = 12;
        params.KernelW = 12;

        cnn::Op::cascade(resized, params, net12, net12c, outputs);
        cnn::Op::nms(outputs, .3f);
    
        backProjectDetections(outputs, factor);
    
        displayResults(image, outputs, "net12");
    
        // 24 net
        filterDetections(image, outputs, Size(24,24), net24, net24c, .1f, .1f);
        cnn::Op::nms(outputs, .3f);

        displayResults(image, outputs, "net24");

        // 48 net
        filterDetections(image, outputs, Size(48,48), net48, net48c, .1f, .1f);
        cnn::Op::nms(outputs, .3f);
    
        displayResults(image, outputs, "net48");
    



/*
        string filename = "../../..//weights/12net.bin";
        string ofilename = filename + ".xml";
        cnn::CNN net12("12net");
        createCNN12(filename, net12);
        FileStorage fs12(ofilename, FileStorage::WRITE);
        fs12 << "cnn" <<  net12;
        fs12.release();

        filename = "../../..//weights/12cnet.bin";
        ofilename = filename + ".xml";
        cnn::CNN net12c("12cnet");
        createCNN12Calibration(filename, net12c);
        FileStorage fs12c(ofilename, FileStorage::WRITE);
        fs12c << "cnn" << net12c;
        fs12c.release();

        filename = "../../..//weights/24net.bin";
        ofilename = filename + ".xml";
        cnn::CNN net24("24net");
        createCNN24(filename, net24);
        FileStorage fs24(ofilename, FileStorage::WRITE);
        fs24 << "cnn" << net24;
        fs24.release();

        filename = "../../..//weights/24cnet.bin";
        ofilename = filename + ".xml";
        cnn::CNN net24c("24cnet");
        createCNN24Calibration(filename, net24c);
        FileStorage fs24c(ofilename, FileStorage::WRITE);
        fs24c << "cnn" << net24c;
        fs24c.release();

        filename = "../../..//weights/48net.bin";
        ofilename = filename + ".xml";
        cnn::CNN net48("48net");
        createCNN48(filename, net48);
        FileStorage fs48(ofilename, FileStorage::WRITE);
        fs48 << "cnn" << net48;
        fs48.release();

        filename = "../../..//weights/48cnet.bin";
        ofilename = filename + ".xml";
        cnn::CNN net48c("48cnet");
        createCNN48Calibration(filename, net48c);
        FileStorage fs48c(ofilename, FileStorage::WRITE);
        fs48c << "cnn" << net48c;
        fs48c.release();
*/


    return 0;
}
