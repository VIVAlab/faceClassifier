


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
#include <limits>
#include "cnn.h"

using namespace cnn;

const  string CNNLabel::NAME       = "name";
const  string CNNLabel::PARAMS     = "params";
const  string CNNLabel::WEIGHTS    = "weights";
const  string CNNLabel::LAYERS     = "layers";
const  string CNNLabel::BIAS       = "bias";
const  string CNNLabel::TYPE       = "type";
const  string CNNLabel::NETWORK    = "network";
const  string CNNLabel::CNN        = "cnn";


const string CNNStringParam::PadH    = "padH";
const string CNNStringParam::PadW    = "padW";
const string CNNStringParam::StrideH = "sH";
const string CNNStringParam::StrideW = "sW";
const string CNNStringParam::KernelW = "kW";
const string CNNStringParam::KernelH = "kH";
const string CNNStringParam::KernelD = "kD";
const string CNNStringParam::NLayers = "nLayers";


const string CNNOpType::CONV    = "conv";
const string CNNOpType::RELU    = "relu";
const string CNNOpType::NORM    = "norm";
const string CNNOpType::SOFTMAX = "softmax";
const string CNNOpType::MAXPOOL = "maxpool";
const string CNNOpType::FC      = "fc";

void CNNLayer::setParams(const map<string, float> &p)
{
    for (map<string,float>::const_iterator it = p.begin(); it != p.end(); it++)
        params[it->first] = it->second;
}
void CNNLayer::setParams(const CNNParam &p)
{
    params[cnn::CNNStringParam::PadH] = p.PadH;
    params[cnn::CNNStringParam::PadW] = p.PadW;
    params[cnn::CNNStringParam::StrideH] = p.StrideH;
    params[cnn::CNNStringParam::StrideW] = p.StrideW;
    
    params[cnn::CNNStringParam::KernelH] = p.KernelH;
    params[cnn::CNNStringParam::KernelW] = p.KernelW;
    params[cnn::CNNStringParam::KernelD] = p.KernelD;
    params[cnn::CNNStringParam::NLayers] = p.NLayers;
}

void CNNLayer::setParam(const string &param, float value)
{
    params[param] = value;
}
void CNNLayer::write(FileStorage &fs) const
{
    fs << "{";
    fs << CNNLabel::TYPE    <<  type;
    fs << CNNLabel::WEIGHTS << "[" ;
    for (size_t i = 0; i < weights.size(); i++)
    {
        fs << weights[i];
    }
    fs <<"]";
    fs << CNNLabel::BIAS << "[";
    for (size_t i = 0; i < bias.size(); i++)
    {
        fs << bias[i];
    }
    fs <<"]";
    fs << CNNLabel::PARAMS << "{";
    
    for (std::map<string,float>::const_iterator it=params.begin(); it!=params.end(); ++it)
        fs << it->first << it->second;
    
    fs << "}";
    fs <<"}";
    
}
void CNNLayer::write(ostream &f) const
{
    cv::writeB(f, type);
    cv::writeB(f, weights);
    cv::writeB(f, bias);
    cv::writeB(f, params);
}

void CNNLayer::read(istream &f)
{
    cv::readB(f, type);
    cv::readB(f, weights);
    cv::readB(f, bias);
    cv::readB(f, params);
}

void CNNLayer::read(const FileNode& node)
{
    weights.clear();
    bias.clear();
    type = (string)node[CNNLabel::TYPE];
    FileNode n = node[CNNLabel::WEIGHTS];
    if (n.type() == FileNode::SEQ)
    {
        FileNodeIterator it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it)
        {
            Mat tmp;
            *it >> tmp;
            weights.push_back(tmp);
        }
    }
    FileNode n2 = node[CNNLabel::BIAS];
    if (n2.type() == FileNode::SEQ)
    {
        FileNodeIterator it = n2.begin(), it_end = n2.end();
        for (; it != it_end; ++it)
        {
            float tmp;
            *it >> tmp;
            bias.push_back(tmp);
        }
    }
    FileNode n3 = node[CNNLabel::PARAMS];
    if (n3.type() == FileNode::MAP)
    {
        FileNodeIterator it = n3.begin(), it_end = n3.end();
        for (; it != it_end; ++it)
        {
            string name = (*it).name();
            int value;
            (*it) >> value;
            params[name] = value;
            
        }
    }
    
}


void CNN::forward(const Mat &input, vector<Mat> &output) const
{
    vector<Mat> _input;
    split(input, _input);
    
    for (size_t i = 0; i < _network.size(); i++)
    {
        const CNNLayer &layer = _layers[_map.at(_network[i])];
        
        bool lastLayer = (i == _network.size() - 1);
        
        vector<Mat> _tmp;
        
        if (layer.type == cnn::CNNOpType::CONV)
        {
            cnn::Op::CONV(_input, layer.weights, _tmp, layer.bias,
                          layer.params.at(cnn::CNNStringParam::NLayers),
                          layer.params.at(cnn::CNNStringParam::KernelD),
                          layer.params.at(cnn::CNNStringParam::StrideW),
                          layer.params.at(cnn::CNNStringParam::StrideH),
                          layer.params.at(cnn::CNNStringParam::PadW),
                          layer.params.at(cnn::CNNStringParam::PadH));
            
        }
        else if (layer.type == cnn::CNNOpType::RELU)
        {
            cnn::Op::RELU(_input, _tmp);
        }
        else if (layer.type == cnn::CNNOpType::SOFTMAX)
        {
            cnn::Op::SOFTMAX(_input, _tmp);
        }
        else if (layer.type == cnn::CNNOpType::MAXPOOL)
        {
            cnn::Op::MAX_POOL(_input, _tmp,
                              layer.params.at(cnn::CNNStringParam::KernelW),
                              layer.params.at(cnn::CNNStringParam::KernelH),
                              layer.params.at(cnn::CNNStringParam::StrideW),
                              layer.params.at(cnn::CNNStringParam::StrideH),
                              layer.params.at(cnn::CNNStringParam::PadW),
                              layer.params.at(cnn::CNNStringParam::PadH));
        }
        else if (layer.type == cnn::CNNOpType::FC)
        {
            cnn::Op::FC(_input, layer.weights, layer.bias,
                        _tmp, layer.params.at(cnn::CNNStringParam::NLayers));
        }
        
        if (_debug)
        {
            cout << _network[i] << endl; 
            for (size_t k = 0; k < _tmp.size(); k++)
            {
                printf("%d %d\n", _tmp[k].rows, _tmp[k].cols);
                cout << _tmp[k] << endl;
            }
        }
        if (lastLayer)
        {
            output = std::move(_tmp);
        }
        else
        {
            _input = std::move(_tmp);
        }
    }
}


string CNN::generateLayerName(const string &type)
{
    size_t layerN = _layers.size();
    string name   = to_string(layerN) + "." +  _name + "." + type;
    return name;
}


CNNLayer& CNN::getLayer(const string &name)
{
    return _layers[_map.at(name)];
}

CNNLayer& CNN::addLayer(const CNNLayer &layer)
{
    size_t layerN = _layers.size();
    string name   = generateLayerName(layer.type);
    
    _map[name] = layerN;
    _layers.push_back(layer);
    _network.push_back(name);
    return _layers[_map.at(name)];
}

void CNN::write(FileStorage &fs) const
{
    fs << "{";
    fs << CNNLabel::NAME << _name;
    fs << CNNLabel::LAYERS << "[";
    for (size_t i = 0; i < _layers.size(); i++)
    {
        fs << _layers[i];
    }
    fs << "]";
    fs << CNNLabel::NETWORK << "[";
    for (size_t i = 0; i < _network.size(); i++)
    {
        fs << _network[i];
    }
    fs << "]";
    fs << "}";
}
void CNN::write(ostream &f) const
{
    cv::writeB(f, _name);
    cv::writeB(f, _layers);
    cv::writeB(f, _network);
    cv::writeB(f, _map);
}

void CNN::read(istream &f)
{
    cv::readB(f, _name);
    cv::readB(f, _layers);
    cv::readB(f, _network);
    cv::readB(f, _map);
}

void CNN::read(const FileNode &node)
{
    _layers.clear();
    _map.clear();
    _network.clear();
    _name = (string)node[CNNLabel::NAME];
    FileNode n = node[CNNLabel::LAYERS];
    if (n.type() == FileNode::SEQ)
    {
        FileNodeIterator it = n.begin(), end = n.end();
        for (; it != end; it++)
        {
            CNNLayer tmp;
            *it >> tmp;
            addLayer(tmp);
        }
    }
}



ostream& cnn::operator<<(ostream &out, const CNNLayer& w)
{
    out << "{ "<< endl;
    out << "\t" << CNNLabel::TYPE << ": " << w.type << endl;
    if (w.bias.size())
    {
        cout << "\t" << CNNLabel::BIAS << ": (" << w.bias.size() << ") [" << w.bias;
        out << "]" << endl;
    }
    if (w.weights.size())
    {
        cout << "\t" << CNNLabel::WEIGHTS << ": ("<< w.weights.size() << ") ["<< endl;
        for (size_t i = 0; i < w.weights.size(); i++)
        {
            out << "\t\t" << w.weights[i].rows<< "x" << w.weights[i].cols;
            if (w.weights[i].channels() > 1)
                out << "x" << w.weights[i].channels() << " ";
            out << w.weights[i].reshape(1,1) << endl;
        }
        out << "\t]" << endl;
    }
    if (w.params.size())
    {
        out << "\t" << CNNLabel::PARAMS << "[" << w.params << "]" << endl;
    }
    out << "}" << endl;
    
    return out;
}

ostream& cnn::operator<<(ostream &out, const CNN& w)
{
    cout << CNNLabel::NAME    << ": \t\t"<< w._name << endl;
    cout << CNNLabel::NETWORK << ": \t"  << w._network << endl;
    cout << CNNLabel::LAYERS  << ": \t[" << w._layers << "]" << endl;
    return out;
}

void Op::bgr2yuv(const Mat &input, Mat &output)
{
    vector<Mat> bgr;
    split(input, bgr);
    vector<Mat> yuv(3);
    for (size_t i = 0; i < yuv.size(); i++)
        yuv[i].create(bgr[i].size(), CV_32F);

        yuv[0] = 0.299   * bgr[2] + 0.587   * bgr[1] + 0.114   * bgr[0];
        yuv[1] = -0.14713* bgr[2] - 0.28886 * bgr[1] + 0.436   * bgr[0];
        yuv[2] = 0.615   * bgr[2] - 0.51499 * bgr[1] - 0.10001 * bgr[0];

        merge(yuv, output);

}
void Op::CONV(const vector<Mat> &input,
              const vector<Mat> &weights,
              vector<Mat> &output,
              const vector<float> &bias,
              const int nLayers,
              const int kernelDepth,
              const int strideH,
              const int strideV,
              const int paddH,
              const int paddV)
{
    output.resize(nLayers);
    vector<Mat> _conv(kernelDepth);
    
    for (size_t i = 0, _inputIdx = 0, _layersInput = 0; i < weights.size(); i++, _inputIdx++)
    {
        
        

        conv(input[_inputIdx], weights[i], _conv[_inputIdx],
             0, strideH, strideV, paddH, paddV);
        
        if (_inputIdx == (kernelDepth - 1))
        {
            output[_layersInput].create(_conv[0].size(), CV_32F);
            _conv[0].copyTo(output[_layersInput]);
            
            for (size_t k = 1; k < _conv.size(); k++)
            {
                output[_layersInput] += _conv[k];
            }
            output[_layersInput] += bias[_layersInput];
            _layersInput++;
            
            _inputIdx = -1;
        }
        
    }
    
}

void Op::MAX_POOL(const vector<Mat> &input,
                  vector<Mat> &output,
                  int width,
                  int height,
                  int strideH ,
                  int strideV ,
                  int paddingH ,
                  int paddingV)
{
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); i++)
    {
        max_pool(input[i], output[i],
                 width, height, strideH, strideV, paddingH, paddingV);
    }
}



void Op::FC(const vector<Mat> &input,
             const vector<Mat> &weights,
             const vector<float> &bias,
             vector<Mat> &output,
             size_t outputs)
{
    Op::CONV(input, weights, output, bias, outputs, weights.size()/outputs, 1, 1, 0, 0);
}


void Op::SOFTMAX(const vector<Mat> &input,
                        vector<Mat> &output)
{
    output.resize(input.size());
    for (size_t k = 0; k < input.size(); k++)
    {
        output[k].create(input[k].size(), input[k].type());
    }
    
    
    for (size_t r = 0; r < input[0].rows; r++)
        for (size_t c = 0; c < input[0].cols; c++)
        {
            Mat _vector(input.size(), 1, CV_32F), _vectorOutput;
            
            for (size_t k = 0; k < input.size(); k++)
            {
                _vector.at<float>(k) = input[k].at<float>(r,c);
            }
            
            softmax(_vector, _vectorOutput);
            
            for (size_t k = 0; k < input.size(); k++)
            {
                output[k].at<float>(r,c) = _vectorOutput.at<float>(k);
            }
        }
}

void Op::RELU(const vector<Mat> &input,
              vector<Mat> &output)
{
    output.resize(input.size());
    
    for (size_t i = 0; i < input.size(); i++)
    {
        threshold(input[i], output[i], 0, 1, THRESH_TOZERO);
    }
    
}

void Op::conv(const Mat &input,
              const Mat &weight,
              Mat &output,
              float bias,
              int strideH,
              int strideV,
              int paddingH,
              int paddingV)
{
    Mat _input;
    copyMakeBorder(input, _input, paddingV, paddingV,
                   paddingH, paddingH, BORDER_CONSTANT,
                   Scalar::all(0));
    int newWidth = ((_input.cols - weight.cols)/strideH) + 1;
    int newHeight= ((_input.rows - weight.rows)/strideV) + 1;
    output.create(Size(newWidth, newHeight), input.type());
    for (size_t row = 0, r = 0; row < newHeight; row++, r+=strideV )
        for (size_t col = 0, c = 0; col < newWidth; col++,  c+= strideH)
        {
            output.at<float>(row, col) = weight.dot(_input(Rect(c, r, weight.cols, weight.rows))) + bias;
        }
    
}

void Op::relu(const Mat &input, Mat &output)
{
    threshold(input, output, 0, 1, THRESH_TOZERO);
}

void Op::softmax(const Mat &input, Mat &output)
{
    output.create(input.size(), CV_32F);
    double _min, _max;
    cv::minMaxIdx(input, &_min, &_max);
    exp(input - _max, output);
    double _sum = sum(output).val[0];
    output = output / _sum;
    
}

void Op::normMeanStd(const Mat &input, Mat &output, const Scalar &mean, const Scalar &stdev)
{
    vector<Mat> layers;
    split(input, layers);
    for (size_t l = 0; l < layers.size(); l++)
    {
        layers[l] = (layers[l]- mean.val[l]);
        layers[l] = layers[l] / ( (stdev.val[l] == 0.f) ? 1 : stdev.val[l] );
    }
    merge(layers, output);
}
void Op::normGlobal(const Mat &input,
                    Mat &output)
{
    Scalar mean, stdev;
    meanStdDev(input, mean, stdev);
    normMeanStd(input, output, mean, stdev);
}

void Op::max_pool(const Mat &input,
                  Mat &output,
                  int width,
                  int height,
                  int strideH,
                  int strideV,
                  int paddingH,
                  int paddingV)
{
    
    Mat _input;
    copyMakeBorder(input, _input, paddingV, paddingV,
                   paddingH, paddingH, BORDER_CONSTANT,
                   Scalar::all(std::numeric_limits<float>::lowest()));
    int newWidth = ((_input.cols - width)/strideH) + 1;
    int newHeight= ((_input.rows - height)/strideV) + 1;
    output.create(Size(newWidth, newHeight), input.type());
    
    for (size_t row = 0, r = 0; row < newHeight; row++, r+=strideV )
        for (size_t col = 0, c = 0; col < newWidth; col++,  c+= strideH)
        {
            double _max;
            
            minMaxIdx(_input(Rect(c, r, width, height)), NULL, &_max);
            
            output.at<float>(row, col) = static_cast<float>(_max);
        } 
}



Detection& cnn::Alg::applyTransformationCode(Detection &detection,
                                          const Mat &response,
                                          const float thr)
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

void cnn::Alg::calibVisualize()
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
                //
                //                        float xnew2 = b.x + x[xi] * b.width * s[si] - (s[si] - 1) * b.width /2 + nwidth;
                //                        float ynew2 = b.y + y[yi] * b.height* s[si] - (s[si] - 1) * b.height/ 2 + nheight;

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


void cnn::Alg::forward(const Mat &img, const cnn::CNN &net, Mat &score, int layer)
{
    vector<Mat> scores;
    net.forward(img, scores);
    score = std::move(scores[layer]);
}

void cnn::Alg::heatMapFromScore(const Mat &score, Mat &heatmap, cv::Size size)
{
    if (size.width == 0 and size.height == 0)
        size = score.size();
    Mat scaled, adjusted;
    resize(score, scaled, size, 0, 0, INTER_CUBIC);
    scaled.convertTo(adjusted, CV_8UC1, 255);
    applyColorMap(adjusted, heatmap, COLORMAP_JET);
}

void cnn::Alg::detect(const Mat &img,
                      const cnn::CNN &net,
                      const cnn::CNNParam &params,
                      vector<Detection> &detections,
                      Mat &score,
                      float thr,
                      float scale)
{
    forward(img, net, score, 0);

    for (size_t r = 0; r < score.rows; r++)
    {
        for (size_t c = 0; c  < score.cols; c++)
        {
            float response = score.at<float>(r,c);

            if (response > thr)
            {
                Rect face(c * scale, r * scale, params.KernelW, params.KernelH );
                Detection det;
                det.face = face;
                det.score = response;
                detections.push_back(det);
            }
        }
    }
}

void cnn::Alg::calibResults(const vector<Mat> &scores, Mat &results)
{
    results.create(scores.size(), 1, CV_32F);
    for (size_t k = 0; k < scores.size(); k++)
    {
        results.at<float>(k) = scores[k].at<float>(0, 0);
    }
}


void cnn::Alg::forwardDetection(const Mat &image,
                             const vector<Detection> &detections,
                             const cnn::CNN &net,
                             const cnn::CNN &calibNet,
                             const cnn::CNNParam &params,
                             vector<Detection> &outputs,
                             float thr, float calibThr, bool useCalibration)
{

    vector<Mat> score;
    Rect imgRoi(0,0,image.cols, image.rows);

    for (unsigned i = 0; i < detections.size(); i++)
    {
        Mat img, imageROI;
        imageROI = image(detections[i].face & imgRoi);

        resize(imageROI, img, Size(params.KernelW, params.KernelH), 0, 0, INTER_AREA);

        net.forward(img, score);

        if (score[0].at<float>(0,0) > thr)
        {
            Detection detect;
            detect.face = detections[i].face;
            detect.score = score[0].at<float>(0,0);

            if (useCalibration)
            {
                vector<Mat> calibOutput;
                Mat transformation;
                calibNet.forward(img, calibOutput);
                calibResults(calibOutput, transformation);
                applyTransformationCode(detect, transformation, calibThr);
            }

            outputs.push_back(detect);
        }
    }
}

void cnn::Alg::calibrate(const Mat &img,
                      const cnn::CNN &net,
                      vector<Detection> &detections,
                      float calibThr)
{
    Rect imgRoi(0,0,img.cols, img.rows);
    for (size_t i = 0; i < detections.size(); i ++)
    {
        vector<Mat> calibOutput;
        Mat transformation;
        net.forward(img(detections[i].face & imgRoi), calibOutput);
        calibResults(calibOutput, transformation);
        applyTransformationCode(detections[i], transformation, calibThr);
    }
}

void cnn::Alg::nms(vector<Detection> &detections,
                const float &threshold)
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

void cnn::Alg::backProject(vector<Detection> &detects,
                        const double &factor)
{
    for (size_t i = 0 ; i < detects.size(); i++)
    {

        detects[i].face.x /= factor;
        detects[i].face.y /= factor;
        detects[i].face.width  /= factor;
        detects[i].face.height /= factor;

    }
}

void cnn::Alg::displayResults(Mat &image,
                           vector<Detection> &detections,
                           const string wName,
                           bool wait,
                           bool save)
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

    if (save)
        imwrite(wName + ".png", tmp);

    if (wait)
        waitKey();
}




