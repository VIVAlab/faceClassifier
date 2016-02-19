#include "storage.h"

using namespace cnn;

const  string CNNLabel::NAME       = "name";
const  string CNNLabel::PARAMS     = "params";
const  string CNNLabel::WEIGHTS    = "weihts";
const  string CNNLabel::LAYERS     = "layers";
const  string CNNLabel::BIAS       = "bias";
const  string CNNLabel::TYPE       = "type";
const  string CNNLabel::NETWORK    = "network";
const  string CNNLabel::CNN        = "cnn";


const string CNNParam::PadH    = "padH";
const string CNNParam::PadW    = "padW";
const string CNNParam::StrideH = "sH";
const string CNNParam::StrideW = "sW";
const string CNNParam::KernelW = "kW";
const string CNNParam::KernelH = "kH";


const string CNNOpType::CONV    = "conv";
const string CNNOpType::RELU    = "relu";
const string CNNOpType::NORM    = "norm";
const string CNNOpType::SOFTMAC = "softmac";
const string CNNOpType::MAXPOOL = "maxpool";
const string CNNOpType::FC      = "fc";

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
    for (auto net : _network)
    {
        CNNLayer &layer = _layers[_map[net]];
        
        InputArray _input = (output.empty())? input: output;
        
        if (layer.type == cnn::CNNOpType::CONV)
        {
            cnn::Op::CONV(_input, layer.weights, output, layer.bias,
                          layer.params[cnn::CNNParam::StrideW],
                          layer.params[cnn::CNNParam::StrideH],
                          layer.params[cnn::CNNParam::PadW],
                          layer.params[cnn::CNNParam::PadH]);
        }
        else if (layer.type == cnn::CNNOpType::RELU)
        {
            cnn::Op::RELU(_input, output);
        }
        else if (layer.type == cnn::CNNOpType::SOFTMAC)
        {
            cnn::Op::SOFTMAC(_input, output);
        }
        else if (layer.type == cnn::CNNOpType::NORM)
        {
            cnn::Op::norm(_input, output);
        }
        else if (layer.type == cnn::CNNOpType::MAXPOOL)
        {
            cnn::Op::MAX_POOL(_input, output,
                              layer.params[cnn::CNNParam::KernelW],
                              layer.params[cnn::CNNParam::KernelH],
                              layer.params[cnn::CNNParam::StrideW],
                              layer.params[cnn::CNNParam::StrideH],
                              layer.params[cnn::CNNParam::PadW],
                              layer.params[cnn::CNNParam::PadH]);
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

void CNN::save(const string &filename)
{
    FileStorage fs(filename, FileStorage::WRITE);
    fs << CNNLabel::CNN << *this;
    fs.release();
}

void CNN::load(const string &filename)
{
    FileStorage fs(filename, FileStorage::READ);
    fs[cnn::CNNLabel::CNN] >> *this;
    fs.release();
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