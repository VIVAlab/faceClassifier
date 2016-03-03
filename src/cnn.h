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



        void forward(InputArray input, OutputArray output);

        
        
        friend ostream& operator<<(ostream &out, const CNN& w);
    };



    
    class Op
    {
    public:

        static void CONV(InputArrayOfArrays input,
                         InputArrayOfArrays weights,
                         OutputArrayOfArrays output,
                         vector<float> &bias,
                         const int nLayers,
                         const int kernelD,
                         const int strideW,
                         const int strideH,
                         const int paddW,
                         const int paddH );

        static void MAX_POOL(InputArrayOfArrays  input,
                             OutputArrayOfArrays output,
                             int width,
                             int height,
                             int strideW ,
                             int strideH ,
                             int paddingW ,
                             int paddingH);

        static void FC(InputArrayOfArrays input,
                       InputArrayOfArrays weights,
                       InputArray bias,
                       OutputArrayOfArrays output,
                       size_t outputs);

        static void RELU(InputArrayOfArrays input,
                         OutputArrayOfArrays output);

        static void SOFTMAX(InputArrayOfArrays input,
                            OutputArrayOfArrays output);

        static void softmax(InputArray input,
                            OutputArray output);

        static void conv(InputArray input,
                         InputArray weights,
                         OutputArray output,
                         float bias = 0,
                         int strideW = 1,
                         int strideH = 1,
                         int paddingW = 0,
                         int paddingH = 0 );

        static void relu(InputArray input, OutputArray output);

        static void norm(InputArray input,
                         OutputArray output,
                         Scalar mean = Scalar::all(0),
                         Scalar stdev=Scalar::all(1));

        static void max_pool(InputArray input,
                             OutputArray output,
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
        
        static Detection& applyTransformationCode(Detection &detection,const Mat &response, const float &thr)
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

        static void cascade(const Mat &image, cnn::CNNParam &params,
                            cnn::CNN &net, cnn::CNN &cnet, vector<Detection> &rect,
                            float threshold = 0.5f,
                            float calibThr  = 0.1f)
        {
            for (size_t r = 0; r < image.rows - params.KernelH; r+= params.StrideH)
            {
                for (size_t c = 0; c < image.cols - params.KernelW; c+= params.StrideW)
                {
                    Rect roi(c,r,params.KernelW, params.KernelH);

                    Mat normImg, netOutput;
                    Op::norm(image(roi), normImg);
                    net.forward(normImg, netOutput);

                    if (netOutput.at<float>(0) > threshold)
                    {

                        Detection detection;
                        detection.face  = roi;
                        detection.score = netOutput.at<float>(0);

                        Mat cnetOutput;
                        cnet.forward(normImg, cnetOutput);
                        applyTransformationCode(detection, cnetOutput, calibThr);
                        rect.push_back(std::move(detection));
                    }
                }
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
                        > threshold)

                    {
                        detections.erase(detections.begin() + j);
                        --j;
                    }
                }
            }
            
        }

        static void backProjectDetections(vector<Detection> &detects, const double &factor)
        {
            for (size_t i = 0 ; i < detects.size(); i++)
            {
                detects[i].face.x /= factor;
                detects[i].face.y /= factor;
                detects[i].face.width  /= factor;
                detects[i].face.height /= factor;
            }
        }

        static void filterDetections(Mat &image, vector<Detection> &detects, Size size,
                                     cnn::CNN &cnn, cnn::CNN &cnet,
                                     float thr = 0.5f ,
                                     float calibThr = 0.1f,
                                     bool calib  = true)
        {
            for (size_t i = 0; i < detects.size(); i++)
            {
                Mat _resizedROI, _normROI, _netOutput;
                Rect imageROI(0,0, image.cols, image.rows);
                resize(image(detects[i].face & imageROI), _resizedROI, size);
                Op::norm(_resizedROI, _normROI);
                cnn.forward(_normROI, _netOutput);

                if (_netOutput.at<float>(0) > thr)
                {
                    detects[i].score = _netOutput.at<float>(0);

                    if (calib)
                    {
                        Mat _cnetOutput;
                        cnet.forward(_normROI, _cnetOutput);
                        detects[i] = applyTransformationCode(detects[i], _cnetOutput, .1f);
                    }
                }
                else
                {
                    detects.erase(detects.begin() + i);
                    i--;
                }
            }
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
