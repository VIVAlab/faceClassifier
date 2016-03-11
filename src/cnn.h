
/**************************************************************************************************
 **************************************************************************************************

 BSD 3-Clause License (https://www.tldrlegal.com/l/bsd3)

 Copyright (c) 2016 Andrés Solís Montero <http://www.solism.ca>, All rights reserved.


 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its contributors
 may be used to endorse or promote products derived from this software
 without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 OF THE POSSIBILITY OF SUCH DAMAGE.

 **************************************************************************************************
 **************************************************************************************************/

#ifndef __cnn__
#define __cnn__

#include <fstream>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "bpersistence.hpp"

using namespace cv;
using namespace std;

namespace cnn {

    struct Detection
    {
        Rect face;
        float score;
    };

    struct CNNLabel
    {
        const static string NAME;
        const static string PARAMS;
        const static string WEIGHTS;
        const static string LAYERS;
        const static string BIAS;
        const static string TYPE;
        const static string NETWORK;
        const static string CNN;
    };

    struct CNNParam
    {
        int PadH;
        int PadW;
        int StrideH;
        int StrideW;
        int KernelW;
        int KernelH;
        int KernelD;
        int NLayers;
    };


    struct CNNStringParam
    {
        const static string PadH;
        const static string PadW;
        const static string StrideH;
        const static string StrideW;
        const static string KernelW;
        const static string KernelH;
        const static string KernelD;
        const static string NLayers;
    };


    struct CNNOpType
    {
        const static string CONV;
        const static string RELU;
        const static string NORM;
        const static string SOFTMAX;
        const static string MAXPOOL;
        const static string FC;
    };

    class CNNLayer
    {
    public:
        CNNLayer(){};
        string            type;
        map<string,float> params;

        vector<Mat>       weights;
        vector<float>     bias;

        void write(FileStorage &fs) const;
        void write(ostream &f) const;

        void read(const FileNode& node);
        void read(istream &in);

        void setParam(const string &param, float value);
        void setParams(const map<string, float> &p);
        void setParams(const CNNParam &p);
        friend ostream& operator<<(ostream &out, const CNNLayer& w);
    };

    struct CNN
    {
    private:
        string             _name;
        map<string,size_t> _map;
        vector<CNNLayer>   _layers;
        vector<string>     _network;
        bool _debug;

        string generateLayerName(const string &type);

    public:
        CNN(const string &name = "", bool debug = false): _name(name), _debug(debug){};
        CNNLayer& getLayer(const string &name);
        CNNLayer& addLayer(const CNNLayer &layer);

        void write(FileStorage &fs) const;
        void write(ostream &f) const;
        void read(istream &f);
        void read(const FileNode &node);

        void forward(const Mat &input, vector<Mat> &output) const;

        friend ostream& operator<<(ostream &out, const CNN& w);
    };

    class Op
    {
    public:

        static void CONV(const vector<Mat> &input,
                         const vector<Mat> &weights,
                         vector<Mat> &output,
                         const vector<float> &bias,
                         const int nLayers,
                         const int kernelD,
                         const int strideW,
                         const int strideH,
                         const int paddW,
                         const int paddH);

        static void MAX_POOL(const vector<Mat> &input,
                             vector<Mat> &output,
                             int width,
                             int height,
                             int strideW ,
                             int strideH ,
                             int paddingW ,
                             int paddingH);
        static void FC(const vector<Mat> &input,
                       const vector<Mat> &weights,
                       const vector<float> &bias,
                       vector<Mat> &output,
                       size_t outputs);

        static void RELU(const vector<Mat> &input,
                         vector<Mat> &output);

        static void SOFTMAX(const vector<Mat> &input,
                             vector<Mat> &output);

        static void softmax(const Mat &input,Mat &output);

        static void conv(const Mat &input,
                         const Mat &weight,
                         Mat &output,
                         float bias = 0,
                         int strideW = 1,
                         int strideH = 1,
                         int paddingW = 0,
                         int paddingH = 0 );

        static void relu(const Mat &input, Mat &output);

        static void normGlobal(const Mat &input, Mat &output);

        static void max_pool(const Mat &input, Mat &output,
                             int width,
                             int height,
                             int strideW = 1 ,
                             int strideH = 1,
                             int paddingW = 0,
                             int paddingH = 0);
    };


    class faceDet
    {
    public:

//        static void refine(const Mat &image, const cnn::CNN &network, const Size &size, vector<Detection> &detections, float thr)
//        {
//            Rect imgRoi(0,0, image.cols, image.rows);
//            for (size_t i = 0; i < detections.size(); i++)
//            {
//                vector<Mat> scores;
//                network.forward(resized, scores);
//                if (scores[0].at<float>(0,0) > thr)
//                {
//                    detections[i].score = scores[0].at<float>(0,0);
//                }
//            }
//        }

        static Detection& applyTransformationCode(Detection &detection,const Mat &response, const float thr)
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

                detection.face.x = detection.face.x - tx * detection.face.width  + (ts - 1) * detection.face.width / 2 / ts;
                detection.face.y = detection.face.y - ty * detection.face.height + (ts - 1) * detection.face.height / 2 / ts;
                
                detection.face.width  /= ts;
                detection.face.height /= ts;
                
                
            }

            return detection;
        }

        static void findFaces(const Mat &img, vector<Detection> &detections, float width, float height, float threshold, float scale , bool smooth)
        {
            Mat image;


            if (smooth)
                medianBlur(img, image, 3);

//            static int i = 0;
//            imwrite(to_string(i) + ".jpg", img*255);
//            imwrite(to_string(i++) + "m.jpg", image*255);


            for (size_t r = 0; r < img.rows; r++)
                for (size_t c = 0; c  < img.cols; c++)
                {
                    float response = 0;

                    if (smooth)
                        response = image.at<float>(r,c);
                    else
                        response = img.at<float>(r,c);


                    if (response > threshold)
                    {
                        Rect face(c * scale, r * scale, width, height );
                        Detection det;
                        det.face = face;
                        det.score = response;
                        detections.push_back(det);
                    }
                }
        }
        static void calibVisualize()
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

                        float xorg = r.x - x[xi] * r.width  + (s[si] -1) * r.width / 2 /s[si];
                        float yorg = r.y - y[yi] * r.height + (s[si] -1) * r.height/ 2 /s[si];
                        
                        float owidth  = r.width / s[si];
                        float oheight = r.height / s[si];
                        
                        
                        
                        Rect r2(xorg, yorg, owidth, oheight);
                        rectangle( a, r2.tl(), r2.br(), Scalar(255,0,0));
                        
                    }
            
            rectangle(a, b.tl(), b.br(), Scalar(255,255,0));
            imshow("Calibration Transforms", a);
            waitKey();
        }
        static void detect(const Mat &img, const cnn::CNN &net, const cnn::CNNParam &params, vector<Detection> &detections, float thr, bool smooth = false, float scale = 2.f)
        {
            vector<Mat> scores;
            net.forward(img, scores);
            findFaces(scores[0], detections, params.KernelW, params.KernelH, thr, scale, smooth);
        }
        
        static void forwardDetection(const Mat &image, const vector<Detection> &detections, const cnn::CNN &net, const cnn::CNN &calibNet, const cnn::CNNParam &params, vector<Detection> &outputs, float thr, float calibThr, bool useCalibration = true)
        {
            Mat imageROI;
            vector<Mat> score;
            Mat img = Mat(params.KernelW, params.KernelH, CV_8UC3);
                        
            for (unsigned i = 0; i < detections.size(); i++)
            {
                Rect imgRoi(0,0,image.cols, image.rows);
                imageROI = image(detections[i].face & imgRoi);
                resize(imageROI, img, img.size(), 0, 0, INTER_AREA);

                net.forward(img, score);
                
                if (score[0].at<float>(0,0) > thr)
                {
                    Detection detect;
                    detect.face = detections[i].face;
                    detect.score = score[0].at<float>(0,0);
                    
                    if (useCalibration)
                    {
                        vector<Mat> calibOutput;
                        calibNet.forward(img, calibOutput);
                        Mat transformation(calibOutput.size(), 1, CV_32F);
                        for (size_t k = 0; k < calibOutput.size(); k++)
                        {
                            transformation.at<float>(k) = calibOutput[k].at<float>(0, 0);
                        }
                        
                        cnn::faceDet::applyTransformationCode(detect, transformation, calibThr);
                    }
                    
                    outputs.push_back(detect);
                }
            }
        }

        static void calibrate(const Mat &img, const cnn::CNN &net, vector<Detection> &detections, float calibThr)
        {
            Rect imgRoi(0,0,img.cols, img.rows);
            for (size_t i = 0; i < detections.size(); i ++)
            {
                vector<Mat> calibOutput;
                net.forward(img(detections[i].face & imgRoi), calibOutput);
                Mat transformation(calibOutput.size(), 1, CV_32F);
                for (size_t k = 0; k < calibOutput.size(); k++)
                {
                    transformation.at<float>(k) = calibOutput[k].at<float>(0, 0);
                }

                cnn::faceDet::applyTransformationCode(detections[i], transformation, calibThr);
            }
        }

        static void nms(vector<Detection> &detections, const float &threshold)
        {
            sort(detections.begin(), detections.end(), [](const Detection &i, const Detection &j)
                 { return i.score > j.score;});


            for (unsigned i = 0; i < detections.size(); i++)
            {
                for (unsigned j = i + 1; j < detections.size(); j++)
                {
                    if (
                        (
                         (float)(detections[i].face & detections[j].face).area() /
                         ( detections[i].face.area() + detections[j].face.area() -
                          (detections[i].face & detections[j].face).area() )
                         )
                        >= threshold)

                    {
                        detections.erase(detections.begin() + j);
                        --j;
                    }
                }
            }


        }

        static void backProject(vector<Detection> &detects, const double &factor)
        {
            for (size_t i = 0 ; i < detects.size(); i++)
            {
                
                detects[i].face.x /= factor;
                detects[i].face.y /= factor;
                detects[i].face.width  /= factor;
                detects[i].face.height /= factor;
                
            }
        }

        static void displayResults(Mat &image, vector<Detection> &detections, const string wName = "default", bool wait = false)
        {
            Mat tmp = image.clone();
            for (size_t i = 0; i < detections.size(); i++)
            {
                rectangle(tmp, detections[i].face.tl(),
                          detections[i].face.br(), Scalar(0,0,255), 1);

                string text = to_string(detections[i].score);
                int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
                double fontScale = .4;
                int thickness = 1;

                int baseline=0;
                Size textSize = getTextSize(text, fontFace,
                                            fontScale, thickness, &baseline);
                //baseline += thickness;

                Point textOrg(detections[i].face.tl().x ,
                              detections[i].face.tl().y );
                
                putText(tmp, text, textOrg, fontFace, fontScale,
                        Scalar(255,0,0), thickness, 8);
                
            }
            imshow(wName, tmp);
//            imwrite(wName + ".jpg", tmp);
            if (wait)
                waitKey();
        }
    };

    static void writeB(ostream &fs, const CNNLayer &layer)
    {
        layer.write(fs);
    }
    static void readB(istream &fs, CNNLayer &layer)
    {
        layer.read(fs);
    }
    static void writeB(ostream &fs, const CNN &cnn)
    {
        cnn.write(fs);
    }
    static void readB(istream &fs, CNN &cnn)
    {
        cnn.read(fs);
    }

    static void write(FileStorage& fs, const string&, const CNNLayer& x)
    {
        x.write(fs);
    }
    static void write(FileStorage& fs, const string&, const CNN& x)
    {
        x.write(fs);
    }
    static void read(const FileNode &node, CNNLayer& x, const cnn::CNNLayer &default_value = cnn::CNNLayer())
    {
        if (node.empty())
            x = default_value;
        else
            x.read(node);
    }
    static void read(const FileNode &node, CNN& x, const cnn::CNN &default_value = cnn::CNN())
    {
        if (node.empty())
            x = default_value;
        else
            x.read(node);
    }
    ostream& operator<<(ostream &out, const CNNLayer& w);
    ostream& operator<<(ostream &out, const CNN& w);

    template<class T> ostream& operator<<(ostream &out, const vector<T> &vec)
    {
        for (size_t i = 0; i < vec.size(); i++)
        {
            out << vec[i] << ((i == vec.size()-1)?"":", ");
        }
        return out;
    }
    template<class K, class V> ostream& operator<<(ostream &out, const map<K,V> &m)
    {
        size_t count = 0;
        for (typename map<K,V>::const_iterator it = m.begin(); it!= m.end(); it++, count++)
        {
            out << it->first << ": " << it->second << ((count<(m.size()-1))? ", ": "");
        }
        return out;
    }
    
    
}






#endif
