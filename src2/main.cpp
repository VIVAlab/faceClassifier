#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
#include <vector>
#include <iostream>
#include <fstream>
#include "storage.h"

void readMats(size_t amount, size_t rows, size_t cols, size_t depth, ifstream &f, vector<Mat> &mats)
{
    vector<float> buffer(amount*depth*rows*cols);
    f.read(reinterpret_cast<char*>(&buffer[0]), amount*rows*cols*depth*sizeof(float));
    mats.resize(amount * depth);
    for (size_t i = 0; i < amount; i++)
    {
        for (size_t j = 0; j < depth; j++){
            mats[i*depth + j] = Mat(rows,cols, CV_32F, &buffer[(i*depth+j)*rows*cols]).t();
        }
    }
}
void readVector(size_t amount, ifstream &f, vector<float> &vector)
{
    vector.resize(amount);
    f.read((char*)&vector[0], amount * sizeof(float));
}

void network_read(const string &filename, const string &name, const string &ofilename)
{
    cnn::CNN net(name);

    cnn::Layer module1, module2, module3, module4, module5, module6, module7;
    module1.type =  cnn::CNNOpType::CONV;

    ifstream f(filename, ios::in | ios::binary);

    readMats(16, 3,3,1, f, module1.weights);
    readVector(16, f, module1.bias);

    module1.setParam(cnn::CNNParam::PadH, 0);
    module1.setParam(cnn::CNNParam::PadW,   0);
    module1.setParam(cnn::CNNParam::StrideW, 1);
    module1.setParam(cnn::CNNParam::StrideH,  1);
    module1.setParam(cnn::CNNParam::KernelH, 3);
    module1.setParam(cnn::CNNParam::KernelW,3);

    net.addLayer(module1);

    module2.type = cnn::CNNOpType::MAXPOOL;
    module2.setParam(cnn::CNNParam::PadH, 1);
    module2.setParam(cnn::CNNParam::PadW,   1);
    module2.setParam(cnn::CNNParam::StrideH, 2);
    module2.setParam(cnn::CNNParam::StrideW,  2);
    module2.setParam(cnn::CNNParam::KernelW, 3);
    module2.setParam(cnn::CNNParam::KernelH,3);

    net.addLayer(module2);

    module3.type = cnn::CNNOpType::RELU;

    net.addLayer(module3);

    module4.type = cnn::CNNOpType::FC;

    readMats(16, 5,5,16, f, module4.weights);
    readVector(16, f, module4.bias);

    module4.setParam(cnn::CNNParam::PadH, 1);
    module4.setParam(cnn::CNNParam::PadW,   1);
    module4.setParam(cnn::CNNParam::StrideH, 2);
    module4.setParam(cnn::CNNParam::StrideW,  2);
    module4.setParam(cnn::CNNParam::KernelW, 3);
    module4.setParam(cnn::CNNParam::KernelH,3);
    net.addLayer(module4);

    module5.type = cnn::CNNOpType::RELU;
    net.addLayer(module5);

    module6.type = cnn::CNNOpType::FC;
    readMats(2, 16,1,1, f, module6.weights);
    readVector(2, f, module6.bias);
    net.addLayer(module6);

    module7.type = cnn::CNNOpType::SOFTMAC;
    net.addLayer(module7);

    f.close();

    FileStorage fs(ofilename, FileStorage::WRITE);
    fs << "cnn" << net;
    fs.release();

}

int main(int, char**)
{

        string filename = "/home/binghao/faceClassifier/preprocess/module.bin";
        string ofilename = filename + ".xml";
        string name = "12cnet";
        network_read(filename, name, ofilename);
        FileStorage fs2;
        cnn::CNN net;
        fs2.open(ofilename, FileStorage::READ);
        fs2["cnn"] >> net;
        cout << net <<endl;
        fs2.release();


    return 0;
}
