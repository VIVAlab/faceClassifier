#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"




void cascade(const Mat &image, cnn::CNNParam &params, cnn::CNN &net, vector<Rect> &rect)
{
    for (size_t r = 0; r < image.rows - params.KernelH; r+= params.StrideH)
    {
        for (size_t c = 0; c < image.cols - params.KernelW; c+= params.StrideW)
        {
            imshow("window",image(Rect(c, r, params.KernelW, params.KernelH)));
            waitKey();
            Mat test;
            Op::norm(image(Rect(c, r, params.KernelW, params.KernelH)), test);
            Mat output;
            net.forward(test, output);
            if (output.at<float>(0) > output.at<float>(1))
                rect.push_back(Rect(c, r, params.KernelW, params.KernelH));

        }
    }
}


struct Detection
{
    Rect face; //x y width height
    float score;
};

void nms(const vector<Detection> &detections, vector<Detection> &outputs)
{
    Rect a, b;
    
    cout << a.area();
    // a.area() + b.area() - (a & b).area();
}

int main(int, char**)
{
    

    
        string filename = "../../../weights/12net.bin";
        string ofilename = filename + ".xml";
        FileStorage fs(ofilename, FileStorage::READ);
        cnn::CNN net12("12net");
        fs["cnn"] >> net12;
        fs.release();


        string image = "../../..//test/img/group1.jpg";
        Mat tmp = imread(image, IMREAD_GRAYSCALE), img, resized;
        resize(tmp, resized, Size(0,0), 12./72., 12./72., INTER_AREA);

        resized.convertTo(img, CV_32F);
        img = img/255.f;





        vector<Rect> outputs;
        cnn::CNNParam params;
        params.StrideH = 4;
        params.StrideW = 4;
        params.KernelH = 12;
        params.KernelW = 12;
        cascade(img, params, net12, outputs);


//
//        string filename = "../../..//weights/12net.bin";
//        string ofilename = filename + ".xml";
//        cnn::CNN net12("12net");
//        createCNN12(filename, net12);
//        FileStorage fs12(ofilename, FileStorage::WRITE);
//        fs12 << "cnn" <<  net12;
//        cout << net12 << endl;
//        fs12.release();
//
//        filename = "../../..//weights/12cnet.bin";
//        ofilename = filename + ".xml";
//        cnn::CNN net12c("12cnet");
//        createCNN12Calibration(filename, net12c);
//        FileStorage fs12c(ofilename, FileStorage::WRITE);
//        fs12c << "cnn" << net12c;
//        fs12c.release();
//
//        filename = "../../..//weights/24net.bin";
//        ofilename = filename + ".xml";
//        cnn::CNN net24("24net");
//        createCNN24(filename, net24);
//        FileStorage fs24(ofilename, FileStorage::WRITE);
//        fs24 << "cnn" << net24;
//        fs24.release();
//
//        filename = "../../..//weights/24cnet.bin";
//        ofilename = filename + ".xml";
//        cnn::CNN net24c("24cnet");
//        createCNN24Calibration(filename, net24c);
//        FileStorage fs24c(ofilename, FileStorage::WRITE);
//        fs24c << "cnn" << net24c;
//        fs24c.release();
//
//        filename = "../../..//weights/48net.bin";
//        ofilename = filename + ".xml";
//        cnn::CNN net48("48net");
//        createCNN48(filename, net48);
//        FileStorage fs48(ofilename, FileStorage::WRITE);
//        fs48 << "cnn" << net48;
//        fs48.release();
//
//        filename = "../../..//weights/48cnet.bin";
//        ofilename = filename + ".xml";
//        cnn::CNN net48c("48cnet");
//        createCNN48Calibration(filename, net48c);
//        FileStorage fs48c(ofilename, FileStorage::WRITE);
//        fs48c << "cnn" << net48c;
//        fs48c.release();



    return 0;
}
