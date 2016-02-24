#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"


Detection& applyTransformationCode(Detection &detection,const Mat &response, const float &thr)
{
    struct _coords {
        size_t s;
        size_t x;
        size_t y;
    };
    vector<_coords> trans;
    for (size_t i = 0; i< response.rows * response.cols ; i++)
    {
        if (response.at<float>(i) > thr)
        {
            _coords t = { i / 9 , (i / 3) % 3, i % 3 };
            trans.push_back(std::move(t));
        }
    }


    vector<float> s = {0.83, 0.91, 1.0, 1.10, 1.21};
    vector<float> x = {-0.17, 0.0, 0.17};
    vector<float> y = {-0.17, 0.0, 0.17};
    
    float ts = 0.f, tx = 0.f, ty = 0.f;

    if (trans.size())
    {
        for (size_t i = 0; i < trans.size(); i++)
        {
            _coords &_tr = trans[i];
            ts += s[_tr.s];
            tx += x[_tr.x];
            ty += y[_tr.y];
        }

        ts /= trans.size();
        tx /= trans.size();
        ty /= trans.size();
        //cout << ts << " " << tx << " " << ty << endl;
        detection.face.x -= ((tx * detection.face.width)/ts);
        detection.face.y -= ((ty * detection.face.height)/ts);
        detection.face.width /= ts;
        detection.face.width /= ts;
    }
    
    return detection;
}

void loadNet(const string &filename, cnn::CNN &net)
{
    FileStorage fs(filename, FileStorage::READ);
    fs["cnn"] >> net;
    fs.release();
}

int main(int, char**)
{

        string image = "../../..//test/img/group1.jpg";
        Mat tmp = imread(image, IMREAD_GRAYSCALE), img;
        tmp.convertTo(tmp, CV_32F);
        tmp = tmp / 255.f;

        vector<string> files = {
                "../../../weights/12net.bin.xml",
                "../../../weights/12cnet.bin.xml",
                "../../../weights/24net.bin.xml"};

        cnn::CNN net12("12net");
        cnn::CNN net12c("12cnet");
        cnn::CNN net24("24net");
        loadNet(files[0], net12);
        loadNet(files[1], net12c);
        loadNet(files[2], net24);

        double winSize = 12.;
        double minFaceSize = 72.;
        double factor = winSize/minFaceSize;
        resize(tmp, img, Size(0,0), factor, factor, INTER_AREA);


        vector<Detection> outputs;
        cnn::CNNParam params;
        params.StrideH = 4;
        params.StrideW = 4;
        params.KernelH = 12;
        params.KernelW = 12;

        cnn::Op::cascade(img, params, net12, outputs);
        cnn::Op::nms(outputs, .3f); //Thresholds for nms

        //apply calibration
        for (size_t i = 0; i < outputs.size(); i++)
        {
            Mat region = img(outputs[i].face), output;
            net12c.forward(region, output);
            outputs[i] = applyTransformationCode(outputs[i], output, .1f);
        }

        for (size_t i = 0; i < outputs.size(); i++)
        {
            outputs[i].face.x /= factor;
            outputs[i].face.y /= factor;
            outputs[i].face.width  /= factor;
            outputs[i].face.height /= factor;

            Mat _img24x24, _tmp, output;
            resize(tmp(outputs[i].face), _img24x24, Size(24,24));
            Op::norm(_img24x24, _tmp);
            net24.forward(_tmp, output);
            if (output.at<float>(0) > output.at<float>(1))
            {
                imshow("asdfadsf", _img24x24);
                cvWaitKey();
                cout <<  "accept "<< endl;
            }
        }






    for (size_t i = 0; i < outputs.size(); i++)
        rectangle(img, outputs[i].face.tl(), outputs[i].face.br(), Scalar::all(255));

        imshow("after", img);
        waitKey();

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
