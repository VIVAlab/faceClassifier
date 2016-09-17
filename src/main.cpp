#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include "storage.h"


#include <ctime>
#include <chrono>


int main(int, char**)
{
        // Read the model .bin files  to .xml
        //cnn::createCNNs();

        // Load networks and modules
        vector<string> files = {
			"../../../weights/model_20net.bin.xml",
			"../../../weights/model_12cnet.bin.xml",
			"../../../weights/model_48net.bin.xml",
			"../../../weights/model_48cnet.bin.xml",
		};

		cnn::CNN net20("20net");
		cnn::CNN net12c("12cnet");
		cnn::CNN net48("48net");
		cnn::CNN net48c("48cnet");
		loadNet(files[0], net20);
		loadNet(files[1], net12c);
		loadNet(files[2], net48);
		loadNet(files[3], net48c);

        // Load image for face detection
        string imageFilename = "../../../test/img/group4.jpg";
        Mat display = imread(imageFilename);
        Mat image = imread(imageFilename, IMREAD_GRAYSCALE), imageN, resized;

        image.convertTo(image, CV_32F);
        image = image / 255.f;

        cnn::Op::normGlobal(image, imageN);

        double winSize = 20.;
        double minFaceSize = 30;
        double maxFaceSize = 180;
        double pyramidRate = sqrt(2.0);
        double faceSize = minFaceSize;
        double factor;

		// first layer
        cnn::CNNParam params;
		params.KernelH = 20;
		params.KernelW = 20;
		vector<cnn::Detection> outputs;
		vector<cnn::Detection> outputs12;
		vector<cnn::Detection> outputs48;

        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
    
    
        while (faceSize < min(image.rows, image.cols) && faceSize < maxFaceSize)
        {
            factor = winSize/faceSize;
            resize(imageN, resized, Size(0,0), factor, factor, INTER_AREA);
            Mat score;

			//cnn::Alg::detect(resized, net20, params, outputs, score, .5f);
			//cnn::Alg::nms(outputs, .1f);
			//cnn::Alg::calibrate(resized, net12c, outputs, 0.1f);
			//cnn::Alg::nms(outputs, .1f);
			//cnn::Alg::backProject(outputs, factor);
			////cnn::Alg::displayResults(display, outputs, "Face Size "+ to_string((int)faceSize));
			//outputs12.insert(outputs12.end(), outputs.begin(), outputs.end());

            cnn::Alg::detect(resized, net20, params, outputs, score, .5f);

            Mat heatmap;
            cnn::Alg::heatMapFromScore(score, heatmap, image.size());
			cnn::Alg::displayResults(resized, outputs, "12net_resized");
			cnn::Alg::backProject(outputs, factor);
			cnn::Alg::displayResults(display, outputs, "12net");
			outputs12.insert(outputs12.end(), outputs.begin(), outputs.end());
            imshow("heatmap", heatmap);
            waitKey();

            faceSize *= pyramidRate;
            outputs.clear();
        }

		cnn::Alg::displayResults(display, outputs12, "12net");

		//params.KernelH = 48;
		//params.KernelW = 48;
		//cnn::Alg::forwardDetection(image, outputs12, net48, net48c, params, outputs48, .99f, .5f, true);
		////cnn::Alg::forwardDetection(image, outputs24, net48, net48c, params, outputs48, .99f, .5f, true);
		//cnn::Alg::nms(outputs48, .1f);
		//cnn::Alg::displayResults(display, outputs48, "results");

        end = std::chrono::system_clock::now();
        std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/ 1000.f) << " seconds" << std::endl;

        waitKey();

        return 0;
}
