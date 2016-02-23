#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"


int main(int, char**)
{

        string filename = "/home/binghao/faceClassifier/weights/12net.bin";
        string ofilename = filename + ".xml";
        cnn::CNN net12("12net");
        createCNN12(filename, net12);
        FileStorage fs12(ofilename, FileStorage::WRITE);
        fs12 << "cnn" <<  net12;
        // cout << net12 << endl;
        fs12.release();

        filename = "/home/binghao/faceClassifier/weights/12cnet.bin";
        ofilename = filename + ".xml";
        cnn::CNN net12c("12cnet");
        createCNN12Calibration(filename, net12c);
        FileStorage fs12c(ofilename, FileStorage::WRITE);
        fs12c << "cnn" << net12c;
        fs12c.release();

        filename = "/home/binghao/faceClassifier/weights/24net.bin";
        ofilename = filename + ".xml";
        cnn::CNN net24("24net");
        createCNN24(filename, net24);
        FileStorage fs24(ofilename, FileStorage::WRITE);
        fs24 << "cnn" << net24;
        fs24.release();

        filename = "/home/binghao/faceClassifier/weights/24cnet.bin";
        ofilename = filename + ".xml";
        cnn::CNN net24c("24cnet");
        createCNN24Calibration(filename, net24c);
        FileStorage fs24c(ofilename, FileStorage::WRITE);
        fs24c << "cnn" << net24c;
        fs24c.release();

        filename = "/home/binghao/faceClassifier/weights/48net.bin";
        ofilename = filename + ".xml";
        cnn::CNN net48("48net");
        createCNN48(filename, net48);
        FileStorage fs48(ofilename, FileStorage::WRITE);
        fs48 << "cnn" << net48;
        fs48.release();

        filename = "/home/binghao/faceClassifier/weights/48cnet.bin";
        ofilename = filename + ".xml";
        cnn::CNN net48c("48cnet");
        createCNN48Calibration(filename, net48c);
        FileStorage fs48c(ofilename, FileStorage::WRITE);
        fs48c << "cnn" << net48c;
        fs48c.release();



    return 0;
}
