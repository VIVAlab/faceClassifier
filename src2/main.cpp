#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"






int main(int, char**)
{
    

    
        string filename = "../../../weights/12net.bin";
        string ofilename = filename + ".xml";
        FileStorage fs(ofilename, FileStorage::READ);
        cnn::CNN net12("12net");
        fs["cnn"] >> net12;
        fs.release();



        filename = "../../../weights/12cnet.bin";
        ofilename = filename + ".xml";
        cnn::CNN net12c("12cnet");
        createCNN12Calibration(filename, net12c);
        FileStorage fs12c(ofilename, FileStorage::READ);
        fs12c["cnn"] >> net12c;
        fs12c.release();



        string image = "../../..//test/img/group1.jpg";
        Mat tmp = imread(image, IMREAD_GRAYSCALE), img, resized;
        resize(tmp, resized, Size(0,0), 12./72., 12./72., INTER_AREA);


        resized.convertTo(img, CV_32F);
        img = img/255.f;





        vector<Detection> outputs;
        cnn::CNNParam params;
        params.StrideH = 4;
        params.StrideW = 4;
        params.KernelH = 12;
        params.KernelW = 12;

        cnn::Op::cascade(img, params, net12, outputs);
        cnn::Op::nms(outputs, .3f);

    for (size_t i = 0; i < outputs.size(); i++)
    {
        Mat region = img(outputs[i].face), output;
        
        net12c.forward(region, output);
        cout << output << endl; 
    }

    Mat copy = img.clone();
    for (size_t i = 0; i < outputs.size(); i++)
        rectangle(copy, outputs[i].face.tl(), outputs[i].face.br(), Scalar::all(255));
    imshow("before", copy);
    cvWaitKey();




    for (size_t i = 0; i < outputs.size(); i++)
        rectangle(img, outputs[i].face.tl(), outputs[i].face.br(), Scalar::all(255));

        imshow("after", img);
        cvWaitKey();

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
