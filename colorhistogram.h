#ifndef COLORHISTOGRAM_H
#define COLORHISTOGRAM_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

class ColorHistogram
{
private:
    int histSize[3];//直方图重箱子的数量
    float hranges[2];//值范围
    const float* ranges[3];//值范围的指针
    int channels[3];//要检查的通道数量

public:
    ColorHistogram(){
        //准备一维直方图的默认参数
        histSize[0] = histSize[1] = histSize[2] = 256;//256个箱子
        hranges[0] = 0.0;//从0开始（包含0）
        hranges[1] = 256.0;//到256结束（不包含256）
        ranges[0] = hranges;//
        ranges[1] = hranges;//
        ranges[2] = hranges;//
        channels[0] = 0;//先关注通道0
        channels[1] = 1;//先关注通道0
        channels[2] = 2;//先关注通道0
    }

    void setSize(int size){
        histSize[0] = histSize[1] = histSize[2] = size;
    }

    Mat getHistogram(const Mat &image);
    Mat getHueHistogram(const Mat &image, int minSaturation = 0);
    SparseMat getSparseHistogram(const Mat &image);

};

#endif // COLORHISTOGRAM_H
