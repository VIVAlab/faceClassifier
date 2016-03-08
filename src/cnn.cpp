


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
const  string CNNLabel::WEIGHTS    = "weihts";
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


void CNN::forward(InputArray input, OutputArray output)
{
    vector<Mat> _input;
    split(input, _input);

    for (size_t i = 0; i < _network.size(); i++)
    {
        CNNLayer &layer = _layers[_map[_network[i]]];

        vector<Mat> _tmp;

        if (layer.type == cnn::CNNOpType::CONV)
        {
            cnn::Op::CONV(_input, layer.weights, _tmp, layer.bias,
                          layer.params[cnn::CNNStringParam::NLayers],
                          layer.params[cnn::CNNStringParam::KernelD],
                          layer.params[cnn::CNNStringParam::StrideW],
                          layer.params[cnn::CNNStringParam::StrideH],
                          layer.params[cnn::CNNStringParam::PadW],
                          layer.params[cnn::CNNStringParam::PadH]);
            
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
                              layer.params[cnn::CNNStringParam::KernelW],
                              layer.params[cnn::CNNStringParam::KernelH],
                              layer.params[cnn::CNNStringParam::StrideW],
                              layer.params[cnn::CNNStringParam::StrideH],
                              layer.params[cnn::CNNStringParam::PadW],
                              layer.params[cnn::CNNStringParam::PadH]);
        }
        else if (layer.type == cnn::CNNOpType::FC)
        {
            cnn::Op::FC(_input, layer.weights, layer.bias, _tmp, layer.params[cnn::CNNStringParam::NLayers]);
        }

        if (_debug)
        {
//            for (size_t i = 0; i < _input.size(); i++)
//            {
//                printf("%d %d\n", _input[i].rows, _input[i].cols);
//                cout << _input[i] <<  endl;
//            }
            for (size_t i = 0; i < _tmp.size(); i++)
            {
                printf("%d %d\n", _tmp[i].rows, _tmp[i].cols);
                cout << _tmp[i] << endl;
            }
        }
        if (i == _network.size() - 1)
        {

            _tmp[0].copyTo(output.getMatRef());
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

void Op::CONV(const vector<Mat> &input,
              const vector<Mat> &weights,
              vector<Mat> &output,
              const vector<float> &bias,
              const int nLayers,
              const int kernelDepth,
              const int strideH,
              const int strideV,
              const int paddH,
              const int paddV )
{
    vector<Mat> _weights, _input;
    input.getMatVector(_input);
    weights.getMatVector(_weights);

    if (output.needed())
        output.create(nLayers, 1, CV_32F);

    vector<Mat> _conv(kernelDepth);
    
    for (size_t i = 0, _inputIdx = 0, _layersInput = 0; i < _weights.size(); i++, _inputIdx++)
    {
        
        conv(_input[_inputIdx], weights.getMat(i), _conv[_inputIdx],
             0, strideH, strideV, paddH, paddV);
        
        
        if (_inputIdx == (kernelDepth - 1))
        {
            Mat &_output = output.getMatRef(_layersInput++);
            _output.create(_conv[0].size(), CV_32F);
            _conv[0].copyTo(_output);
            for (size_t k = 1; k < _conv.size(); k++)
            {
                _output += _conv[k];
            }
            _output += bias[i];
            
          
            _inputIdx = -1;
        }
        
    }

}

void Op::MAX_POOL(InputArrayOfArrays  input,
                  OutputArrayOfArrays output,
                  int width,
                  int height,
                  int strideH ,
                  int strideV ,
                  int paddingH ,
                  int paddingV)
{

    vector<Mat> _input;
    input.getMatVector(_input);

    if (output.needed())
        output.create(_input.size(),1, CV_32F);

    for (size_t i = 0; i < _input.size(); i++)
    {
        max_pool(_input[i], output.getMatRef(i),
                 width, height, strideH, strideV, paddingH, paddingV);
    }
}


void Op::FC(InputArrayOfArrays input,
            InputArrayOfArrays weights,
            InputArray bias,
            OutputArrayOfArrays output,
            size_t outputs)
{

    vector<Mat> _input;
    input.getMatVector(_input);

    if (output.needed())
        output.create(1, 1, CV_32F);

    Mat &_output = output.getMatRef(0);
    _output.create(outputs, 1, CV_32F);
    Mat _bias   = bias.getMat();

    for (size_t o_index = 0; o_index < outputs; o_index++)
    {
        double sum = 0;
        for (size_t i_index = 0, w_index = o_index * _input.size(); i_index < _input.size(); i_index++, w_index++)
        {
            Mat tmp;
            multiply(input.getMat(i_index), weights.getMat(w_index), tmp);
            sum+= cv::sum(tmp)[0];
        }
        _output.at<float>(o_index) = sum + _bias.at<float>(o_index);
    }
}
void Op::SOFTMAX(InputArrayOfArrays input,
                 OutputArrayOfArrays output)
{
    vector<Mat> _input;
    input.getMatVector(_input);

    if (output.needed())
        output.create(_input.size(),1, CV_32F);

    for (size_t i = 0; i < _input.size(); i++)
    {
        softmax(input.getMat(i), output.getMatRef(i));
    }
}

void Op::RELU(InputArrayOfArrays input,
              OutputArrayOfArrays output)
{
    vector<Mat> _input;
    input.getMatVector(_input);

    if (output.needed())
        output.create(_input.size(),1, CV_32F);

    for (size_t i = 0; i < _input.size(); i++)
    {
        threshold(input.getMat(i), output.getMatRef(i), 0, 1, THRESH_TOZERO);
    }

}

void Op::conv(InputArray input,
              InputArray weights,
              OutputArray output,
              float bias ,
              int strideH ,
              int strideV,
              int paddingH ,
              int paddingV )
{

    Mat _input, _weight, _output;
    _weight = weights.getMat();

    if (paddingH || paddingV)
        copyMakeBorder(input, _input, paddingV, paddingV,
                       paddingH, paddingH, BORDER_CONSTANT,
                       Scalar::all(0));
    else
        _input = input.getMat();

    int newWidth = ((_input.cols - _weight.cols)/strideH) + 1;
    int newHeight= ((_input.rows - _weight.rows)/strideV) + 1;
    output.create(Size(newWidth, newHeight), input.type());
    _output = output.getMat();

    for (size_t row = 0; row < newHeight; row+=strideV )
        for (size_t col = 0; col < newWidth; col += strideH)
        {
            _output.at<float>(col, row) = _weight.dot(_input(Rect(row, col,
                                                                  _weight.cols, _weight.rows))) + bias;
        }
}

void Op::relu(InputArray input, OutputArray output)
{
    threshold(input, output, 0, 1, THRESH_TOZERO);
}

void Op::softmax(InputArray input,
                 OutputArray output)
{


    Mat _input = input.getMat();
    output.create(_input.size(), CV_32F);
    Mat &_output= output.getMatRef();

    double _min, _max;
    cv::minMaxIdx(_input, &_min, &_max);

    exp(_input - _max, _output);

    double _sum = sum(_output).val[0];
    _output = _output / _sum;

}

void Op::norm(InputArray input,
              OutputArray output,
              Scalar mean,
              Scalar stdev)
{
    Scalar _mean, _stdev;
    meanStdDev(input, _mean, _stdev);
    vector<Mat> layers;
    split(input, layers);
    for (size_t l = 0; l < layers.size(); l++)
    {
        layers[l] = (layers[l]-(_mean.val[l] + mean.val[l]));

        if (_stdev.val[l] == 0.f)
            _stdev.val[l] = 1.f;

        layers[l] = layers[l] * (stdev.val[l]/_stdev.val[l]);
    }
    merge(layers, output);
}

void Op::normGlobal(InputArray input,
                    OutputArray output)
{
    Scalar mean, stdev;
    meanStdDev(input, mean, stdev);
    vector<Mat> layers;
    split(input, layers);
    for (size_t l = 0; l < layers.size(); l++)
    {
        layers[l] = (layers[l]- mean.val[l]);

        if (stdev.val[l] == 0.f)
            stdev.val[l] = 1.f;

        layers[l] = layers[l] / stdev.val[l];
    }
    merge(layers, output);
}

void Op::max_pool(InputArray input,
                  OutputArray output,
                  int width,
                  int height,
                  int strideH,
                  int strideV,
                  int paddingH,
                  int paddingV)
{

    Mat _input, _output;

    if (paddingH || paddingV)
        copyMakeBorder(input, _input, paddingV, paddingV,
                       paddingH, paddingH, BORDER_CONSTANT,
                       Scalar::all(std::numeric_limits<float>::lowest()));
    else
        _input = input.getMat();

    int newWidth = ((_input.cols - width)/strideH) + 1;
    int newHeight= ((_input.rows - height)/strideV) + 1;
    output.create(Size(newWidth, newHeight), input.type());
    _output = output.getMat();

    for (size_t row = 0, r = 0; row < newHeight; row++, r+=strideV )
        for (size_t col = 0, c = 0; col < newWidth; col++,  c+= strideH)
        {
            double _max;

            minMaxIdx(_input(Rect(r, c, width, height)), NULL, &_max);

            _output.at<float>(col, row) = static_cast<float>(_max);
        } 
}
