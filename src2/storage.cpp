#include "storage.h"

using namespace cnn;

string Layer::NAME     = "name";
string Layer::PARAMS   = "params";
string Layer::PARAM    = "param";
string Layer::WEIGHTS  = "weights";
string Layer::BIAS     = "bias";
string Layer::TYPE     = "type";
string CNN::NAME       = "name";
string CNN::LAYERS     = "layers";
string CNN::NETWORK    = "network";

void Layer::setParam(OPTION param, float value)
{
    params[(int)param] = value;
}
void Layer::write(FileStorage &fs) const
{
    fs << "{";
    fs << NAME   << name;
    fs << TYPE   <<  type;
    fs << WEIGHTS << "[" ;
    for (size_t i = 0; i < weights.size(); i++)
    {
        fs << weights[i];
    }
    fs <<"]";
    fs << BIAS << "[";
    for (size_t i = 0; i < bias.size(); i++)
    {
        fs << bias[i];
    }
    fs <<"]";
    fs << PARAMS << "{";
    
    for (std::map<int,float>::const_iterator it=params.begin(); it!=params.end(); ++it)
        fs << Layer::PARAM+to_string(it->first) << it->second;
    
    fs << "}";
    fs <<"}";
    
}
void Layer::read(const FileNode& node)
{
    weights.clear();
    bias.clear();
    name = (string)node[NAME];
    type = (string)node[TYPE];
    FileNode n = node[WEIGHTS];
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
    FileNode n2 = node[BIAS];
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
    FileNode n3 = node[PARAMS];
    if (n3.type() == FileNode::MAP)
    {
        FileNodeIterator it = n3.begin(), it_end = n3.end();
        for (; it != it_end; ++it)
        {
            string name = (*it).name();
            int key = atoi(name.substr(Layer::PARAM.size()).c_str());
            int value;
            (*it) >> value;
            params[key] = value ;
            
        }
    }
    
}



Layer& CNN::getLayer(const string &name)
{
    return _layers[_map.at(name)];
}

Layer& CNN::addLayer(const Layer &layer)
{
    _map[layer.name] = _layers.size();
    _layers.push_back(layer);
    _network.push_back(layer.name);
    return _layers[_map.at(layer.name)];
}

void CNN::write(FileStorage &fs) const
{
    fs << "{";
    fs << NAME << _name;
    fs << LAYERS << "[";
    for (size_t i = 0; i < _layers.size(); i++)
    {
        fs << _layers[i];
    }
    fs << "]";
    fs << NETWORK << "[";
    for (size_t i = 0; i < _network.size(); i++)
    {
        fs << _network[i];
    }
    fs << "]";
    fs << "}";
}
void CNN::read(const FileNode &node)
{
    _layers.clear();
    _map.clear();
    _network.clear();
    _name = (string)node[NAME];
    FileNode n = node[LAYERS];
    if (n.type() == FileNode::SEQ)
    {
        FileNodeIterator it = n.begin(), end = n.end();
        for (; it != end; it++)
        {
            Layer tmp;
            *it >> tmp;
            _map[tmp.name] = _layers.size();
            _layers.push_back(tmp);
        }
    }
    FileNode n2 = node[NETWORK];
    if (n2.type() == FileNode::SEQ)
    {
        FileNodeIterator it = n2.begin(), end = n2.end();
        for (; it != end; it++)
        {
            string tmp;
            *it >> tmp;
            _network.push_back(tmp);
        }
    }
}

Ptr<CNN> CNN::loadCNNFromFile(const string& filename)
{
    Ptr<CNN> tmp = new CNN("");
    FileStorage fs(filename, FileStorage::READ);
    fs[NAME] >> *tmp;
    fs.release();
    return tmp;
}

void CNN::writeCNNToFile(const CNN &cnn, const string &filename)
{
    FileStorage fs(filename, FileStorage::WRITE);
    fs << NAME << cnn;
    fs.release();
}



ostream& cnn::operator<<(ostream &out, const Layer& w)
{
    out << "{ "<< endl << "\t" << Layer::NAME << ": " << w.name << endl;
    out << "\t" << Layer::TYPE << ": " << w.type << endl;
    if (w.bias.size())
    {
        cout << "\t" << Layer::BIAS << ": (" << w.bias.size() << ") [";
        for (size_t i = 0; i < w.bias.size(); i++)
            out << w.bias[i] << ((i!=w.bias.size()-1)?", ":"");
        out << "]" << endl;
    }
    if (w.weights.size())
    {
        cout << "\t" << Layer::WEIGHTS << ": ("<< w.weights.size() << ") ["<< endl;
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
        out << "\t" << Layer::PARAMS << "[";
        for (std::map<int,float>::const_iterator it=w.params.begin(); it!=w.params.end(); ++it)
            cout << Layer::PARAM << it->first <<": "<< it->second << ((it!=(--w.params.end()))?", ":"");
        out << "]" << endl;
    }
    out << "}" << endl;
    
    return out;
}
ostream& cnn::operator<<(ostream &out, const CNN& w)
{
    cout << CNN::NAME << ": \t\t" << w._name << endl;
    cout << CNN::NETWORK << ": \t";
    for (size_t i = 0; i < w._network.size(); i++)
    {
        out << w._network[i] << ((i<w._network.size()-1)?" -> ":"") ;
    }
    out << endl;
    out << CNN::LAYERS << ": \t[" << endl;
    for (size_t i = 0; i < w._layers.size(); i++)
    {
        out <<  w._layers[i];
        if (i != w._layers.size()- 1)
        {
            out << ", " << endl;
        }
    }
    out << "]" << endl;
    return out;
}