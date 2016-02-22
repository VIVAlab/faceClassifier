#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"


int main(int, char**)
{

        string filename = "/home/binghao/faceClassifier/preprocess/module.bin";
        string ofilename = filename + ".xml";
        cnn::CNN net("12cnet");
        createCNN12(filename, net);

        FileStorage fs(ofilename, FileStorage::WRITE);
        fs << "cnn" <<  net;
        cout << net <<endl;
        fs.release();


    return 0;
}
