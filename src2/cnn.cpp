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

#include "cnn.h"

using namespace cnn;

void Op::CONV(InputArray input,
                 InputArrayOfArrays weights,
                 OutputArrayOfArrays output,
                 vector<float> bias,
                 int strideH,
                 int strideV,
                 int paddH,
                 int paddV )
{
    CV_Assert(weights.size().width == 1 || weights.size().height == 1);
    size_t output_items = max(weights.size().width, weights.size().height);

    if (output.needed())
    {
        output.create(output_items,1 , CV_32F);
    }

    for (size_t i = 0; i < output_items; i++)
    {
        conv(input, weights.getMat(i), output.getMatRef(i),
             bias[i], strideH, strideV, paddH, paddV);
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
    CV_Assert(input.size().width == 1 || input.size().height == 1);
    size_t input_items = max(input.size().width, input.size().height);

    if (output.needed())
        output.create(input_items,1, CV_32F);

    for (size_t i = 0; i < input_items; i++)
    {
        max_pool(input.getMat(i), output.getMatRef(i),
                 width, height, strideH, strideV, paddingH, paddingV);
    }
}


void Op::FC(InputArrayOfArrays input,
            InputArrayOfArrays weights,
            InputArray bias,
            OutputArray output,
            size_t outputs)
{

    CV_Assert(input.size().width == 1 || input.size().height == 1);
    size_t input_items = max(input.size().width, input.size().height);

    output.create(outputs, 1, CV_32F);
    Mat _output = output.getMat();
    Mat _bias   = bias.getMat();

    for (size_t o_index = 0; o_index < outputs; o_index++)
    {
        double sum = 0;
        for (size_t i_index = 0, w_index = o_index * input_items; i_index < input_items; i_index++, w_index++)
        {
            Mat tmp;
            multiply(input.getMat(i_index), weights.getMat(w_index), tmp);
            sum+= cv::sum(tmp)[0];
        }
        _output.at<float>(o_index) = sum + _bias.at<float>(o_index);
    }
}


void Op::RELU(InputArrayOfArrays input,
              OutputArrayOfArrays output)
{
    CV_Assert(input.size().width == 1 || input.size().height == 1);
    size_t input_items = max(input.size().width, input.size().height);

    if (output.needed())
        output.create(input_items,1, CV_32F);

    for (size_t i = 0; i < input_items; i++)
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
    CV_Assert((input.type()   == CV_32F) &&
              (weights.type() == CV_32F) &&
              (input.channels() == weights.channels()));

    Mat border, kernel, convul;
    if (paddingH || paddingV)
    {
        copyMakeBorder(input, border, paddingV, paddingV,
                       paddingH, paddingH,
                       BORDER_CONSTANT,
                       Scalar::all(0));
    }
    else
    {
        border = input.getMat();
    }

    kernel = weights.getMat();

    int newWidth = ((input.cols() - weights.cols() + 2*paddingH)/strideH) + 1;
    int newHeight= ((input.rows() - weights.rows() + 2*paddingV)/strideV) + 1;
    output.create(Size(newWidth, newHeight), input.type());
    convul = output.getMat();

    int channels = border.channels();

    float *rI,*rK,*rO;

    for (size_t row = 0; row < newHeight; row+= strideV) //rows
    {
        rI = border.ptr<float>(row);
        rO = convul.ptr<float>(row);
        for (size_t col = 0; col < newWidth; col+= strideH) //columns
        {
            size_t inputRealCol = channels * col;
            float sum = 0.f;

            for (size_t c  = 0; c  < channels; c++)
            {
                for (size_t krow = 0; krow < weights.rows(); krow++) //kernel rows
                {
                    rK = kernel.ptr<float>(krow);
                    for (size_t kcol = 0; kcol < weights.cols(); kcol++) //kernel columns
                    {
                        sum += rI[inputRealCol + c] * rK[channels * kcol + c];

                    }
                }
            }
            sum += bias;
            rO[col] = sum;
        }
    }

}

void Op::relu(InputArray input, OutputArray output)
{
    threshold(input, output, 0, 1, THRESH_TOZERO);
}

void Op::SOFTMAC(InputArray input,
                 OutputArray output)
{
    output.create(input.size(), CV_32F);
    Mat _input = input.getMat();

    double _min, _max;
    cv::minMaxLoc(_input, &_min, &_max);

    Mat _output= output.getMat();
    _output = _input - _max;
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
        layers[l] = (layers[l]-(_mean.val[l] + mean.val[l]))*stdev.val[l]/_stdev.val[l];
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
    CV_Assert(input.channels() == 1 && input.type() == CV_32F);

    Mat border, pooling;
    if (paddingH || paddingV)
    {
        copyMakeBorder(input, border, paddingV, paddingV,
                       paddingH, paddingH,
                       BORDER_CONSTANT,
                       Scalar::all(0));
    }
    else
    {
        border = input.getMat();
    }

    int newWidth = ((input.cols() - width  + 2*paddingH)/strideH) + 1;
    int newHeight= ((input.rows() - height + 2*paddingV)/strideV) + 1;

    output.create(Size(newWidth, newHeight), input.type());
    pooling = output.getMat();


    float *rI,*rO;

    for (size_t row = 0; row < newHeight; row+= strideV) //rows
    {
        rI = border.ptr<float>(row);
        rO = pooling.ptr<float>(row);
        for (size_t col = 0; col < newWidth; col+= strideH) //columns
        {
            float max = rI[col];

            for (size_t prow = 0; prow < height; prow++) //kernel rows
            {
                rI = border.ptr<float>(row + prow);
                for (size_t pcol = 0; pcol < width; pcol++) //kernel columns
                {
                    if (rI[pcol] > max)
                        max = rI[pcol];

                }
            }

            rO[col] = max;
        }
    }

}
