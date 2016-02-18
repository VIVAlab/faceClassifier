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

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace cnn {



    class Op
    {

    public:
        


    static void CONV(InputArray input,
                     InputArrayOfArrays weights,
                     OutputArrayOfArrays output,
                     vector<float> bias,
                     int strideW,
                     int strideH,
                     int paddW,
                     int paddH );

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
                   OutputArray output,
                   size_t outputs);

    static void RELU(InputArrayOfArrays input,
                     OutputArrayOfArrays output);
    static void SOFTMAC(InputArray input,
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



}






#endif
