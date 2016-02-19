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

#ifndef __BIN__STORAGE__H
#define __BIN__STORAGE__H

#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;

namespace cv {
    
class BFileStorage
{

public:
    enum class Mode { READ, WRITE };
    enum { UNDEFINED = 0, OPENED = 1};
    
    BFileStorage(BFileStorage&)   = delete;
    void operator=(BFileStorage&) = delete;
    
    BFileStorage(const string &filename, Mode flags): _mode(flags)
    {
        if (flags == Mode::READ)
        {
            _fin.open(filename, ios::in | ios::binary);
            _status = _fin.is_open();
        }
        if (flags == Mode::WRITE)
        {
            _fout.open(filename, ios::out | ios::binary);
            _status = _fout.is_open();
        }
    }
    bool isOpened()
    {
        return _status;
    }
    void release()
    {
        if (!isOpened())
            return;
        if (_mode == Mode::READ)
            _fin.close();
        else if (_mode == Mode::WRITE)
            _fout.close();
    }
    
    template<typename _Tp>
    friend BFileStorage& operator << (BFileStorage& bfs, const _Tp& value);
    template<typename _Tp>
    friend BFileStorage& operator >> (BFileStorage& bfs, _Tp& value);
    
private:
    ifstream _fin;
    ofstream _fout;
    Mode _mode;
    int _status;
    
};
  
template<typename _Tp>
BFileStorage& operator << (BFileStorage& fs, const _Tp& value)
{
    if (fs.isOpened() && fs._mode == BFileStorage::Mode::WRITE)
    writeB(fs._fout, value);
    return fs;
}
    
template<typename _Tp>
BFileStorage& operator >> (BFileStorage& fs, _Tp& value)
{
    if (fs.isOpened() && fs._mode == BFileStorage::Mode::READ)
    readB(fs._fin, value);
    return fs;
}
    


template<typename T> static inline void writeScalar(ostream& out, const T& data) {
    out.write(reinterpret_cast<const char*>(&data), sizeof(T));
}

template<typename T> static inline void readScalar(istream& in, T& data) {
    in.read(reinterpret_cast<char*>(&data), sizeof(T));
}

static inline void writeB(ostream& out, const size_t &_data)
{
    writeScalar(out, _data);
}
static inline void writeB(ostream& out, const int &_data)
{
    writeScalar(out, _data);
}
static inline void writeB(ostream& out, const float &_data)
{
    writeScalar(out, _data);
}
static inline void writeB(ostream& out, const double &_data)
{
    writeScalar(out, _data);
}
static inline void writeB(ostream& out, const string &_data)
{
    writeScalar(out, _data);
}

static inline void readB(istream& in, size_t &_data)
{
    readScalar(in, _data);
}

static inline void readB(istream& in, int &_data)
{
    readScalar(in, _data);
}
static inline void readB(istream& in, float &_data)
{
    readScalar(in, _data);
}
static inline void readB(istream& in, double &_data)
{
    readScalar(in, _data);
}
static inline void readB(istream& in, string &_data)
{
    readScalar(in, _data);
}


static inline void writeB(ostream& out, const Mat &_data)
{
    writeScalar(out, _data.rows);
    writeScalar(out, _data.cols);
    writeScalar(out, _data.type());
    const size_t bytes = _data.cols * _data.elemSize();
    for (int i = 0; i < _data.rows; i++)
        out.write(reinterpret_cast<const char*>(_data.ptr(i, 0)), bytes);
}

static inline void readB(istream& in, Mat &_data)
{
    int rows, cols, type;
    readScalar(in, rows);
    readScalar(in, cols);
    readScalar(in, type);
    _data.create(rows, cols, type);
    const size_t bytes = _data.cols * _data.elemSize();
    for (int i = 0; i < _data.rows; i++)
        in.read(reinterpret_cast<char*>(_data.ptr(i, 0)), bytes);
}


template<typename T> static inline void writeB(ostream& out, const vector<T>& vec) {
    writeScalar(out, vec.size());
    for (typename  vector<T>::const_iterator it = vec.begin(); it!=vec.end(); it++)
        writeB(out, *it);
}

template<typename K, typename V> static inline void writeB(ostream& out, const map<K,V> &dict)
{
    writeScalar(out, dict.size());
    for (typename map<K,V>::const_iterator it=dict.begin(); it!=dict.end(); ++it)
    {
        writeB(out, it->first);
        writeB(out, it->second);
    }
}
template<typename T> static inline void writeB(ostream& out, const vector<vector<T>> &vec)
{
    writeScalar(out, vec.size());
    for (typename vector<vector<T>>::const_iterator it = vec.begin(); it!=vec.end(); it++)
        writeB(out, *it);
}


template<typename T> static inline void readB(istream& in, vector<T>& vec) {
    size_t size;
    readScalar(in, size);
    vec.resize(size);
    for (size_t i = 0; i < vec.size(); i++)
        readB(in, vec[i]);
}
template<typename K, typename V> static inline void readB(istream& in,map<K,V> &dict)
{
    size_t size;
    readScalar(in, size);
    for (size_t i = 0; i < size; i++)
    {
        K key;
        V value;
        readB(in, key);
        readB(in, value);
        dict.insert(pair<K,V>(key,value));
    }
}
template<typename T> static inline void readB(istream& in, vector<vector<T>> &vec)
{
    size_t size;
    readScalar(in, size);
    vec.resize(size);
    for (typename vector<vector<T>>::iterator it; it != vec.end(); it++)
        readB(in, *it);
}
    
}

#endif 
