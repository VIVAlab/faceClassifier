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

#define BINARY false

namespace cnn
{
    void loadNet(const string &filename, cnn::CNN &net, bool binary = BINARY);
    void saveNet(const string &filename, cnn::CNN &net, bool binary = BINARY);
    void readMats(size_t amount, size_t rows, size_t cols, size_t depth, ifstream &f, vector<Mat> &mats);
    void readVector(size_t amount, ifstream &f, vector<float> &vector);
    void createCNNLayer(cnn::CNNLayer &layer, const string &type, const CNNParam &params,  ifstream *file = nullptr);
    void createMAXPOOL(cnn::CNNLayer &layer,  const CNNParam &params);
    void createCONV(cnn::CNNLayer &layer,  const CNNParam &params, ifstream &file);
    void createFC(cnn::CNNLayer &layer,   const CNNParam &params, ifstream &file);
    void createRELU(cnn::CNNLayer &layer);
    void createSOFTMAX(cnn::CNNLayer &layer);


    static void createCNN20x16(const string &filename, cnn::CNN &net)
    {
        ifstream f(filename, ios::in | ios::binary);

        cnn::CNNLayer module1, module2, module3, module4, module5, module6, module7;

        CNNParam params;
        params.PadH = 1;
        params.PadW = 1;
        params.StrideW = 2;
        params.StrideH = 2;
        params.KernelH = 3;
        params.KernelW = 3;
        params.KernelD = 3;
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
        params.KernelW = 4;
        params.KernelD = 16;
        params.NLayers = 16;
        createFC(module4, params, f);
        net.addLayer(module4);

        createRELU(module5);
        net.addLayer(module5);

        params.KernelH = 1;
        params.KernelW = 1;
        params.KernelD = 16;
        params.NLayers = 2;
        createFC(module6, params, f);
        net.addLayer(module6);

        createSOFTMAX(module7);
        net.addLayer(module7);

        f.close();
    }
    static void createUpperBodyCNNs()
    {
        vector<string> files = {
            "../../../weights/20x16net.bin"
        };
        string extXML = ".xml";

        cnn::CNN net20x16("20x16net");

        createCNN20x16(files[0], net20x16);

        saveNet(files[0] + extXML, net20x16);

    }

}






#endif
