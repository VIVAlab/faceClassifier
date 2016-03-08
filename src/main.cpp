#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"



void displayResults(Mat &image, vector<Detection> &detections, const string wName = "default")
{



    Mat tmp = image.clone();
    for (size_t i = 0; i < detections.size(); i++)
    {
        rectangle(tmp, detections[i].face.tl(),
                  detections[i].face.br(), Scalar::all(255));

        string text = to_string(detections[i].score);
//        std::cout << i << ": " << detections[i].score << "\t\t|\t" <<
//                                  detections[i].face.x<< "\t" <<
//                                  detections[i].face.y<< " \t" <<
//                                  detections[i].face.width << "\t"<<
//                                  detections[i].face.height << endl;
        int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontScale = .4;
        int thickness = 1;

        int baseline=0;
        Size textSize = getTextSize(text, fontFace,
                                    fontScale, thickness, &baseline);
        baseline += thickness;

        Point textOrg(detections[i].face.tl().x ,
                      detections[i].face.tl().y );

        putText(tmp, text, textOrg, fontFace, fontScale,
                Scalar::all(255), thickness, 8);

    }
    imshow(wName, tmp);
    // waitKey();
}


void calibVisualize()
{
    Mat a = Mat::zeros(500,500, CV_8UC3);
    const Rect b(100, 100,100,100);

    vector<float> s = {0.83, 0.91, 1.0, 1.10, 1.21};
    vector<float> x = {-0.17, 0.0, 0.17};
    vector<float> y = {-0.17, 0.0, 0.17};


    for (size_t si = 0; si < s.size(); si++)
        for (size_t xi = 0; xi < x.size(); xi++)
            for (size_t yi = 0; yi < y.size(); yi++)
            {
                float xnew = b.x + x[xi] * b.width * s[si] - (s[si] - 1) * b.width /2;
                float ynew = b.y + y[yi] * b.height* s[si] - (s[si] - 1) * b.height/ 2;

                float nwidth  =  b.width * s[si];
                float nheight =  b.height * s[si];


                float xnew2 = b.x + x[xi] * b.width * s[si] - (s[si] - 1) * b.width /2 + nwidth;
                float ynew2 = b.y + y[yi] * b.height* s[si] - (s[si] - 1) * b.height/ 2 + nheight;

                Rect r(xnew, ynew, nwidth, nheight);
                rectangle( a, r.tl(), r.br(), Scalar::all(255));

                //invert

                //This is my program entry point
                // and nwidth and nheight are the sizes of the detections
                // xnew and ynew are the detection coordinates of the top left corner

                nwidth = xnew2 - xnew ;
                nheight= ynew2 - ynew ;

                float xorg = xnew - x[xi] * nwidth  + (s[si] -1) * nwidth / 2 /s[si];
                float yorg = ynew - y[yi] * nheight + (s[si] -1) * nheight/ 2 /s[si];

                float owidth  = nwidth / s[si];
                float oheight = nheight / s[si];



                Rect r2(xorg, yorg, owidth, oheight);
                rectangle( a, r2.tl(), r2.br(), Scalar(255,0,0));

            }

    rectangle(a, b.tl(), b.br(), Scalar(255,255,0));
    imshow("transformations", a);
    cvWaitKey();
}


int main(int, char**)
{
        // read .bin to .xml
//        binToXML();

        string imageFilename = "../../../test/img/group1.jpg";

        Mat image = imread(imageFilename, IMREAD_GRAYSCALE), display, imageN, resized;
        image.convertTo(image, CV_32F);
        image = image / 255.f;
        display = image.clone();

        Op::normGlobal(image, imageN);

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
        cnn::CNN net48("48net");
        cnn::CNN net48c("48cnet");
        loadNet(files[0], net12);
        loadNet(files[1], net12c);
        loadNet(files[2], net24);
        loadNet(files[3], net24c);
        loadNet(files[4], net48);
        loadNet(files[5], net48c);

        double winSize = 12.;
        double minFaceSize = 48;
        double pyramidRate = 1.414;
        double faceSize = minFaceSize;
        double factor;
    
        vector<Detection> outputs;
        vector<Detection> g_outputs;
        cnn::CNNParam params;
        params.StrideH = 4;
        params.StrideW = 4;
        params.KernelH = 12;
        params.KernelW = 12;


    //calibVisualize();

    while (faceSize < min(image.rows, image.cols))
    {
        factor = winSize/faceSize;
        
        // 12 net
        resize(imageN, resized, Size(0,0), factor, factor, INTER_AREA);

        cnn::faceDet::cascade(resized, params, net12, net12c, outputs, 0.5f, .1f, false);
    
        cnn::faceDet::backProjectDetections(outputs, factor);
        cnn::faceDet::nms(outputs, .2f);
        displayResults(display, outputs, "net12");
    
        // 24 net
        cnn::faceDet::filterDetections(image, outputs, Size(24,24), net24, net24c, .0000001f, .1f);
        cnn::faceDet::nms(outputs, .5f);
        displayResults(image, outputs, "net24");

        // 48 net
        cnn::faceDet::filterDetections(image, outputs, Size(48,48), net48, net48c, .1f, .1f);
        displayResults(image, outputs, "net48");

        g_outputs.insert(g_outputs.end(), outputs.begin(), outputs.end());
        
        outputs.clear();
        faceSize *= pyramidRate;
        
        waitKey();
        destroyAllWindows();
    }
    
        // global nms
        cnn::faceDet::nms(g_outputs, .3f);

        displayResults(image, g_outputs, "net48");
        waitKey();

        return 0;
}
