#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
#include <vector>
#include <iostream>
#include <fstream>
#include "storage.h"

void readMats(size_t amount, size_t rows, size_t cols, size_t depth, ifstream &f, vector<Mat> &mats)
{
    Ptr<float> buffer = new float[amount*depth*rows*cols];
    f.read((char*)((float*)buffer), amount*rows*cols*depth*sizeof(float));
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
    module1.type =  "conv";
    module1.name =  name + module1.type;

    ifstream f(filename, ios::in | ios::binary);

    readMats(16, 3,3,1, f, module1.weights);
    readVector(16, f, module1.bias);

    module1.setParam(cnn::OPTION::HORIZONTAL_PADDING, 0);
    module1.setParam(cnn::OPTION::VERTICAL_PADDING,   0);
    module1.setParam(cnn::OPTION::HORIZONTAL_STRIDE, 1);
    module1.setParam( cnn::OPTION::VERTICAL_STRIDE,  1);
    module1.setParam(cnn::OPTION::KERNEL_WIDTH, 3);
    module1.setParam(cnn::OPTION::KERNEL_HEIGHT,3);

    net.addLayer(module1);

    module2.type = "maxpool";
    module2.name = name + module2.type;
    module2.setParam(cnn::OPTION::HORIZONTAL_PADDING, 1);
    module2.setParam(cnn::OPTION::VERTICAL_PADDING,   1);
    module2.setParam(cnn::OPTION::HORIZONTAL_STRIDE, 2);
    module2.setParam(cnn::OPTION::VERTICAL_STRIDE,  2);
    module2.setParam(cnn::OPTION::KERNEL_WIDTH, 3);
    module2.setParam(cnn::OPTION::KERNEL_HEIGHT,3);

    net.addLayer(module2);

    module3.type = "relu";
    module3.name = name + module3.type;

    net.addLayer(module3);

    module4.type = "fc";
    module4.name = name + module4.type;

    readMats(16, 5,5,16, f, module4.weights);
    readVector(16, f, module4.bias);

    module4.setParam(cnn::OPTION::HORIZONTAL_PADDING, 1);
    module4.setParam(cnn::OPTION::VERTICAL_PADDING,   1);
    module4.setParam(cnn::OPTION::HORIZONTAL_STRIDE, 2);
    module4.setParam(cnn::OPTION::VERTICAL_STRIDE,  2);
    module4.setParam(cnn::OPTION::KERNEL_WIDTH, 3);
    module4.setParam(cnn::OPTION::KERNEL_HEIGHT,3);

    net.addLayer(module4);

    module5.type = "relu";
    module5.name = name + module5.type;

    net.addLayer(module5);

    module6.type = "fc";
    module6.name = name + module6.type + to_string(2);
    readMats(2, 16,1,1, f, module6.weights);
    readVector(2, f, module6.bias);

    net.addLayer(module6);

    module7.type = "softmax";
    module7.name = name + module7.type;

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



        //    vector<float> input = {1,2,3,4,5};
//    vector<float> input2 = {6,7,8,9,10};
//    vector<Mat> a;
//    a.push_back(Mat(input));
//    a.push_back(Mat(input2));
//    Mat tmp;
//    multiply(a, a, tmp);
//
//    cout << tmp <<  endl;






    return 0;
}
