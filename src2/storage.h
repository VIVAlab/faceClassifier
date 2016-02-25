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

#ifndef __storage__
#define __storage__

#include <map>
#include <vector>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "cnn.h"

using namespace std;
using namespace cv;
using namespace cnn;

static void readMats(size_t amount, size_t rows, size_t cols, size_t depth, ifstream &f, vector<Mat> &mats)
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
static void readVector(size_t amount, ifstream &f, vector<float> &vector)
{
    vector.resize(amount);
    f.read((char*)&vector[0], amount * sizeof(float));
}

static void createCNNLayer(cnn::CNNLayer &layer, const string &type, const CNNParam &params,  ifstream *file = nullptr)
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
static void createMAXPOOL(cnn::CNNLayer &layer,  const CNNParam &params)
{
    createCNNLayer(layer, cnn::CNNOpType::MAXPOOL, params);
}
static void createCONV(cnn::CNNLayer &layer,  const CNNParam &params, ifstream &file)
{
    createCNNLayer(layer, cnn::CNNOpType::CONV, params, &file);
}
static void createFC(cnn::CNNLayer &layer,   const CNNParam &params, ifstream &file)
{
    createCNNLayer(layer, cnn::CNNOpType::FC, params, &file);
}
static void createRELU(cnn::CNNLayer &layer)
{
    layer.type = cnn::CNNOpType::RELU;
}
static void createSOFTMAX(cnn::CNNLayer &layer)
{
    layer.type = cnn::CNNOpType::SOFTMAX;
}
static void createCNN12(const string &filename, cnn::CNN &net)
{
    ifstream f(filename, ios::in | ios::binary);

    cnn::CNNLayer module1, module2, module3, module4, module5, module6, module7;

    CNNParam params;
    params.PadH = 0;
    params.PadW = 0;
    params.StrideW = 1;
    params.StrideH = 1;
    params.KernelH = 3;
    params.KernelW = 3;
    params.KernelD = 1;
    params.NLayers = 16;

    createCONV(module1, params, f);
    net.addLayer(module1);

    params.PadH = 1;
    params.PadW = 1;
    params.StrideW = 2;
    params.StrideH = 2;
    params.KernelH = 3;
    params.KernelW = 3;
    params.KernelD = 1;
    params.NLayers = 1;

    createMAXPOOL(module2, params);     // default parameters??
    net.addLayer(module2);

    createRELU(module3);
    net.addLayer(module3);

    params.KernelH = 5;
    params.KernelW = 5;
    params.KernelD = 16;
    params.NLayers = 16;
    createFC(module4, params, f);
    net.addLayer(module4);

    createRELU(module5);
    net.addLayer(module5);

    params.KernelH = 16;
    params.KernelW = 1;
    params.KernelD = 1;
    params.NLayers = 2;
    createFC(module6, params, f);
    net.addLayer(module6);

    createSOFTMAX(module7);
    net.addLayer(module7);

    f.close();
}

static void createCNN12Calibration(const string &filename, cnn::CNN &net)
{
    ifstream f(filename, ios::in | ios::binary);

    cnn::CNNLayer module1, module2, module3, module4, module5, module6, module7;

    CNNParam params;
    params.PadH = 0;
    params.PadW = 0;
    params.StrideW = 1;
    params.StrideH = 1;
    params.KernelH = 3;
    params.KernelW = 3;
    params.KernelD = 1;
    params.NLayers = 16;

    createCONV(module1, params, f);
    net.addLayer(module1);

    createRELU(module2);
    net.addLayer(module2);

    params.PadH = 1;
    params.PadW = 1;
    params.StrideW = 2;
    params.StrideH = 2;
    params.KernelH = 3;
    params.KernelW = 3;
    params.KernelD = 1;
    params.NLayers = 1;
    createMAXPOOL(module3, params);
    net.addLayer(module3);

    params.KernelH = 5;
    params.KernelW = 5;
    params.KernelD = 16;
    params.NLayers = 128;
    createFC(module4, params, f);
    net.addLayer(module4);

    createRELU(module5);
    net.addLayer(module5);

    params.KernelH = 128;
    params.KernelW = 1;
    params.KernelD = 1;
    params.NLayers = 45;
    createFC(module6, params, f);
    net.addLayer(module6);

    createSOFTMAX(module7);
    net.addLayer(module7);

    f.close();
}

static void createCNN24(const string &filename, cnn::CNN &net)
{
    ifstream f(filename, ios::in | ios::binary);

    cnn::CNNLayer module1, module2, module3, module4, module5, module6, module7;

    CNNParam params;
    params.PadH = 0;
    params.PadW = 0;
    params.StrideW = 1;
    params.StrideH = 1;
    params.KernelH = 5;
    params.KernelW = 5;
    params.KernelD = 1;
    params.NLayers = 64;

    createCONV(module1, params, f);
    net.addLayer(module1);

    createRELU(module2);
    net.addLayer(module2);

    params.PadH = 1;
    params.PadW = 1;
    params.StrideW = 2;
    params.StrideH = 2;
    params.KernelH = 3;
    params.KernelW = 3;
    params.KernelD = 1;
    params.NLayers = 1;
    createMAXPOOL(module3, params);
    net.addLayer(module3);

    params.KernelH = 10;
    params.KernelW = 10;
    params.KernelD = 64;
    params.NLayers = 128;
    createFC(module4, params, f);
    net.addLayer(module4);

    createRELU(module5);
    net.addLayer(module5);

    params.KernelH = 128;
    params.KernelW = 1;
    params.KernelD = 1;
    params.NLayers = 2;
    createFC(module6, params, f);
    net.addLayer(module6);

    createSOFTMAX(module7);
    net.addLayer(module7);

    f.close();
}

static void createCNN24Calibration(const string &filename, cnn::CNN &net)
{
    ifstream f(filename, ios::in | ios::binary);

    cnn::CNNLayer module1, module2, module3, module4, module5, module6, module7;

    CNNParam params;
    params.PadH = 0;
    params.PadW = 0;
    params.StrideW = 1;
    params.StrideH = 1;
    params.KernelH = 5;
    params.KernelW = 5;
    params.KernelD = 1;
    params.NLayers = 32;

    createCONV(module1, params, f);
    net.addLayer(module1);

    params.PadH = 1;
    params.PadW = 1;
    params.StrideW = 2;
    params.StrideH = 2;
    params.KernelH = 3;
    params.KernelW = 3;
    params.KernelD = 1;
    params.NLayers = 1;
    createMAXPOOL(module2, params);
    net.addLayer(module2);

    createRELU(module3);
    net.addLayer(module3);

    params.KernelH = 10;
    params.KernelW = 10;
    params.KernelD = 32;
    params.NLayers = 64;
    createFC(module4, params, f);
    net.addLayer(module4);

    createRELU(module5);
    net.addLayer(module5);

    params.KernelH = 64;
    params.KernelW = 1;
    params.KernelD = 1;
    params.NLayers = 45;
    createFC(module6, params, f);
    net.addLayer(module6);

    createSOFTMAX(module7);
    net.addLayer(module7);

    f.close();
}

static void createCNN48(const string &filename, cnn::CNN &net)
{
    ifstream f(filename, ios::in | ios::binary);

    cnn::CNNLayer module1, module2, module3, module4, module5, module6, module7, module8, module9, module10;

    CNNParam params;
    params.PadH = 0;
    params.PadW = 0;
    params.StrideW = 1;
    params.StrideH = 1;
    params.KernelH = 5;
    params.KernelW = 5;
    params.KernelD = 1;
    params.NLayers = 64;

    createCONV(module1, params, f);
    net.addLayer(module1);

    params.PadH = 1;
    params.PadW = 1;
    params.StrideW = 2;
    params.StrideH = 2;
    params.KernelH = 3;
    params.KernelW = 3;
    params.KernelD = 1;
    params.NLayers = 1;
    createMAXPOOL(module2, params);
    net.addLayer(module2);

    createRELU(module3);
    net.addLayer(module3);

    params.PadH = 0;
    params.PadW = 0;
    params.StrideW = 1;
    params.StrideH = 1;
    params.KernelH = 5;
    params.KernelW = 5;
    params.KernelD = 64;
    params.NLayers = 64;

    createCONV(module4, params, f);
    net.addLayer(module4);

    params.PadH = 1;
    params.PadW = 1;
    params.StrideW = 2;
    params.StrideH = 2;
    params.KernelH = 3;
    params.KernelW = 3;
    params.KernelD = 1;
    params.NLayers = 1;
    createMAXPOOL(module5, params);
    net.addLayer(module5);

    createRELU(module6);
    net.addLayer(module6);

    params.KernelH = 9;
    params.KernelW = 9;
    params.KernelD = 64;
    params.NLayers = 256;
    createFC(module7, params, f);
    net.addLayer(module7);

    createRELU(module8);
    net.addLayer(module8);

    params.KernelH = 256;
    params.KernelW = 1;
    params.KernelD = 1;
    params.NLayers = 2;
    createFC(module9, params, f);
    net.addLayer(module9);

    createSOFTMAX(module10);
    net.addLayer(module10);

    f.close();
}

static void createCNN48Calibration(const string &filename, cnn::CNN &net)
{
    ifstream f(filename, ios::in | ios::binary);

    cnn::CNNLayer module1, module2, module3, module4, module5, module6, module7, module8, module9;

    CNNParam params;
    params.PadH = 0;
    params.PadW = 0;
    params.StrideW = 1;
    params.StrideH = 1;
    params.KernelH = 5;
    params.KernelW = 5;
    params.KernelD = 1;
    params.NLayers = 64;

    createCONV(module1, params, f);
    net.addLayer(module1);

    params.PadH = 1;
    params.PadW = 1;
    params.StrideW = 2;
    params.StrideH = 2;
    params.KernelH = 3;
    params.KernelW = 3;
    params.KernelD = 1;
    params.NLayers = 1;
    createMAXPOOL(module2, params);
    net.addLayer(module2);

    createRELU(module3);
    net.addLayer(module3);

    params.PadH = 0;
    params.PadW = 0;
    params.StrideW = 1;
    params.StrideH = 1;
    params.KernelH = 5;
    params.KernelW = 5;
    params.KernelD = 64;
    params.NLayers = 64;

    createCONV(module4, params, f);
    net.addLayer(module4);

    createRELU(module5);
    net.addLayer(module5);

    params.KernelH = 18;
    params.KernelW = 18;
    params.KernelD = 64;
    params.NLayers = 256;
    createFC(module6, params, f);
    net.addLayer(module6);

    createRELU(module7);
    net.addLayer(module7);

    params.KernelH = 256;
    params.KernelW = 1;
    params.KernelD = 1;
    params.NLayers = 45;
    createFC(module8, params, f);
    net.addLayer(module8);

    createSOFTMAX(module9);
    net.addLayer(module9);

    f.close();
}


static void binToXML()
{
    string filename = "../../..//weights/12net.bin";
    string ofilename = filename + ".xml";
    cnn::CNN net12("12net");
    createCNN12(filename, net12);
    FileStorage fs12(ofilename, FileStorage::WRITE);
    fs12 << "cnn" <<  net12;
    cout << net12 << endl;
    fs12.release();

    filename = "../../..//weights/12cnet.bin";
    ofilename = filename + ".xml";
    cnn::CNN net12c("12cnet");
    createCNN12Calibration(filename, net12c);
    FileStorage fs12c(ofilename, FileStorage::WRITE);
    fs12c << "cnn" << net12c;
    fs12c.release();

    filename = "../../..//weights/24net.bin";
    ofilename = filename + ".xml";
    cnn::CNN net24("24net");
    createCNN24(filename, net24);
    FileStorage fs24(ofilename, FileStorage::WRITE);
    fs24 << "cnn" << net24;
    fs24.release();

    filename = "../../..//weights/24cnet.bin";
    ofilename = filename + ".xml";
    cnn::CNN net24c("24cnet");
    createCNN24Calibration(filename, net24c);
    FileStorage fs24c(ofilename, FileStorage::WRITE);
    fs24c << "cnn" << net24c;
    fs24c.release();

    filename = "../../..//weights/48net.bin";
    ofilename = filename + ".xml";
    cnn::CNN net48("48net");
    createCNN48(filename, net48);
    FileStorage fs48(ofilename, FileStorage::WRITE);
    fs48 << "cnn" << net48;
    fs48.release();

    filename = "../../..//weights/48cnet.bin";
    ofilename = filename + ".xml";
    cnn::CNN net48c("48cnet");
    createCNN48Calibration(filename, net48c);
    FileStorage fs48c(ofilename, FileStorage::WRITE);
    fs48c << "cnn" << net48c;
    fs48c.release();
}


#endif
