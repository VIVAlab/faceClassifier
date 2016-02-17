#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
#include "storage.h"
#include "cnn.h"
    



#include <vector>
int main(int, char**)
{
    
//    vector<float> input = {1,2,3,4,5};
//    vector<float> input2 = {6,7,8,9,10};
//    vector<Mat> a;
//    a.push_back(Mat(input));
//    a.push_back(Mat(input2));
//    Mat tmp;
//    multiply(a, a, tmp);
//
//    cout << tmp <<  endl;
    
    
    
    cnn::CNN net("12net");
    
    
    cnn::Layer cnnw1;
    cnnw1.type = "conv";
    cnnw1.name = "12net.conv";

    vector<float> bias = {1., 2., 3.4 , 5, 0.1, 0.1, .23 , .05};
    vector<Mat> weights;
    weights.push_back(Mat::ones(3, 3, CV_32F)*.1);
    weights.push_back(Mat::ones(3, 3, CV_32F)*.3);
    weights.push_back(Mat::ones(3, 3, CV_32F)*.4);
    weights.push_back(Mat::ones(3, 3, CV_32F)*.5);
    weights.push_back(Mat::ones(3, 3, CV_32F)*.1);
    weights.push_back(Mat::ones(3, 3, CV_32F)*.3);
    weights.push_back(Mat::ones(3, 3, CV_32F)*.4);
    weights.push_back(Mat::ones(3, 3, CV_32F)*.5);

    cnnw1.bias = bias;
    cnnw1.weights = weights;
    
    cnnw1.setParam(cnn::OPTION::HORIZONTAL_PADDING, 1);
    cnnw1.setParam(cnn::OPTION::VERTICAL_PADDING, 1);
    cnnw1.setParam(cnn::OPTION::HORIZONTAL_STRIDE, 1);
    cnnw1.setParam( cnn::OPTION::VERTICAL_STRIDE, 1);
    
    
    cnn::Layer cnnw2;
    cnnw2.type = "maxpool";
    cnnw2.name = "12net.maxpool";
    
    net.addLayer(cnnw1);
    net.addLayer(cnnw2);
    
    FileStorage fs("testing.xml", FileStorage::WRITE);
    fs << "layers" << net;
    fs.release();
    
    FileStorage fs2;
    fs2.open("testing.xml", FileStorage::READ);
    fs2["layers"] >> net;
    cout << net <<endl;
    fs2.release();
    
    
    
    return 0;
}