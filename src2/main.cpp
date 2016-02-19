#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
#include <vector>
#include <iostream>
#include <fstream>
#include "storage.h"
#include "bin_storage.h"

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

    cnn::CNNLayer module1, module2, module3, module4, module5, module6, module7;
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
    
    vector<float>  a = {1,3,3,4,5,1,3,53,3}, b;
    map<string, size_t> dict, dict2;
    
    dict.insert(pair<string,size_t>("hello",3));
    dict.insert(pair<string,size_t>("hello2",4));
    dict.insert(pair<string,size_t>("hello43",5));
    dict.insert(pair<string,size_t>("hello5",6));
    string filename = "hello.bin";
    ofstream f(filename, ios::out | ios::binary);
    write(f, dict);
    f.close();

    ifstream f2(filename, ios::in | ios::binary);
    read(f2, dict2);
    f2.close();
    
    for (size_t i = 0; i < b.size(); i++)
        cout << b[i] << endl;
    
    for ( map<string, size_t>::const_iterator it=dict2.begin(); it!=dict2.end(); ++it)
    {
        cout << it->first << " " << it->second <<endl;
    }
//    ReadVariable(f2, filenamer);
//        f2.close();
//    cout << filenamer <<  " " << i2 << endl;
    
//        string filename = "/home/binghao/faceClassifier/preprocess/module.bin";
//        string ofilename = filename + ".xml";
//        string name = "12cnet";
//        network_read(filename, name, ofilename);
//        FileStorage fs2;
//        cnn::CNN net;
//        fs2.open(ofilename, FileStorage::READ);
//        fs2["cnn"] >> net;
//        cout << net <<endl;
//        fs2.release();


    return 0;
}
