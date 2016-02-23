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
#include "opencv2/opencv.hpp"
#include "bpersistence.hpp"

using namespace cv;
using namespace std;

namespace cnn {



    class Op
    {
    public:
    static void CONV(InputArray input,
                     InputArrayOfArrays weights,
                     OutputArrayOfArrays output,
                     vector<float> &bias,
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
    static void SOFTMAX(InputArray input,
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

        string generateLayerName(const string &type);

    public:
        CNN(const string &name = ""): _name(name){};
        CNNLayer& getLayer(const string &name);
        CNNLayer& addLayer(const CNNLayer &layer);

        void write(FileStorage &fs) const;
        void write(ostream &f) const;
        void read(istream &f);
        void read(const FileNode &node);



        void forward(InputArray input, OutputArray output);

        void load(const string &filename);
        void save(const string &filename);


        friend ostream& operator<<(ostream &out, const CNN& w);
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
