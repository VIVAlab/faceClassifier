#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

struct Detection
{
    Rect face;
    float score;
};

void merge(Detection* detection, int lo, int mi, int hi)
{
    Detection* A = detection + lo;

    unsigned lb = mi - lo; Detection* B = new Detection[lb];
    for (unsigned i = 0; i < lb; i++) B[i] = A[i];
    unsigned lc = hi - mi; Detection* C = detection + mi;

    for (unsigned i = 0, j = 0, k = 0; (j < lb) || (k < lc);)
    {
        if ((j < lb) && (!(k < lc) || (B[j].score >= C[k].score))) A[i++] = B[j++];
        if ((k < lc) && (!(j < lb) || (C[k].score >= B[j].score))) A[i++] = C[k++];
    }

    delete [] B;
}

void mergeSort(Detection* detection, unsigned lo, unsigned hi)
{
    if (hi - lo < 2) return;
    unsigned mi = (lo + hi) / 2;
    mergeSort(detection, lo, mi); mergeSort(detection, mi, hi);
    merge(detection , lo, mi, hi);
}

void nms(vector<Detection> &detections, float &threshold)
{
    mergeSort(&detections[0], 0, detections.size());

    float tmp;   
    for (unsigned i = 0; i < detections.size(); i++)
    {
        for (unsigned j = i + 1; j < detections.size(); j++)
        {
            cout << "comparing " << i << " and " << j << ": " << endl;
            if (
                  (
                      tmp = (float)(detections[i].face & detections[j].face).area() /
                      ( detections[i].face.area() + detections[j].face.area() -
                      (detections[i].face & detections[j].face).area() ) 
                  ) 
                  > threshold)
            
            {
                detections.erase(detections.begin() + j);
                --j;
            }
            cout << tmp << endl;
        }
        cout << endl << "vector size: " << detections.size() << endl << endl;
    }
    
    
    for (int i = 0; i < detections.size(); i++)
    {
        cout << detections[i].face << endl;
        cout << detections[i].score << endl << endl;
    }
}

int main(void)
{
    vector<Detection> detection(5);
    vector<Detection> output;

    detection[0].face = Rect(1, 1, 5, 5);
    detection[0].score = 0.55;

    detection[1].face = Rect(2, 3, 7, 7);
    detection[1].score = 0.34;

    detection[2].face = Rect(4, 3, 6, 6);
    detection[2].score = 0.98;
    
    detection[3].face = Rect(5, 5, 2, 2);
    detection[3].score = 0.86;
    
    detection[4].face = Rect(6, 4, 6, 5);
    detection[4].score = 1.00;
    
    float threshold = 0.5;
    nms(detection, threshold);
    
    
    return 0;
}
