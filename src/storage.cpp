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

#include "storage.h"


void cnn::loadNet(const string &filename, cnn::CNN &net, bool binary)
{
    if (binary)
    {
        BFileStorage fs(filename + ".bin", BFileStorage::Mode::READ);
        fs >> net;
        fs.release();
    }
    else
    {
        FileStorage fs(filename, FileStorage::READ);
        fs["cnn"] >> net;
        fs.release();
    }
}
void cnn:: saveNet(const string &filename, cnn::CNN &net, bool binary)
{
    if (binary)
    {
        BFileStorage fs(filename + ".bin", BFileStorage::Mode::WRITE);
        fs << net;
        fs.release();
    }
    else
    {
        FileStorage fs(filename, FileStorage::WRITE);
        fs << "cnn" <<  net;
        fs.release();
    }
};
void cnn::readMats(size_t amount, size_t rows, size_t cols, size_t depth, ifstream &f, vector<Mat> &mats)
{
    vector<float> buffer(amount*depth*rows*cols);
    f.read(reinterpret_cast<char*>(&buffer[0]), amount*rows*cols*depth*sizeof(float));
    mats.resize(amount * depth);
    for (size_t i = 0; i < amount; i++)
    {
        for (size_t j = 0; j < depth; j++){
            mats[i*depth + j]  = Mat(rows, cols, CV_32F, &buffer[(i*depth+j)*rows*cols]).clone();
        }
    }
}
void cnn::readVector(size_t amount, ifstream &f, vector<float> &vector)
{
    vector.resize(amount);
    f.read((char*)&vector[0], amount * sizeof(float));
}
void cnn::createCNNLayer(cnn::CNNLayer &layer, const string &type, const CNNParam &params,  ifstream *file)
{
    layer.type = type;
    layer.setParams(params);
    if (file != nullptr)
    {
        readMats(params.NLayers,
                 params.KernelH,
                 params.KernelW,
                 params.KernelD,
                 *file, layer.weights);
        readVector(params.NLayers,
                   *file, layer.bias);
    }
}
void cnn::createMAXPOOL(cnn::CNNLayer &layer,  const CNNParam &params)
{
    createCNNLayer(layer, cnn::CNNOpType::MAXPOOL, params);
}
void cnn::createCONV(cnn::CNNLayer &layer,  const CNNParam &params, ifstream &file)
{
    createCNNLayer(layer, cnn::CNNOpType::CONV, params, &file);
}
void cnn::createFC(cnn::CNNLayer &layer,   const CNNParam &params, ifstream &file)
{
    createCNNLayer(layer, cnn::CNNOpType::FC, params, &file);
}
void cnn::createRELU(cnn::CNNLayer &layer)
{
    layer.type = cnn::CNNOpType::RELU;
}
void cnn::createSOFTMAX(cnn::CNNLayer &layer)
{
    layer.type = cnn::CNNOpType::SOFTMAX;
}

