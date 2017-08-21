#ifndef CONTENTFINDER_H
#define CONTENTFINDER_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

class ContentFinder
{
private:
    float hranges[2];
    const float* ranges[3];
    int channels[3];

    float threshold;//判断阀值
    Mat histogram;//输入直方图
public:
    ContentFinder():threshold(0.1f){
        ranges[0] = hranges;
        ranges[1] = hranges;
        ranges[2] = hranges;
    }

    //设置直方图的阀值[0,1]
    void setThreshold(float t){
        threshold = t;
    }

    //
    float getThreshold(){
        return threshold;
    }

    //设置引用的直方图
    void setHistogram(const Mat &h){
        histogram = h;
        normalize(histogram, histogram, 1.0);
    }

    Mat find(const Mat &image);
    Mat find(const Mat &image, float minValue, float maxValue, int *channels);
};

#endif // CONTENTFINDER_H
