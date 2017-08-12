#ifndef HISTOGRAM1D_H
#define HISTOGRAM1D_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

//创建灰度图像的直方图
class Histogram1D
{
private:
    int histSize[1];//直方图重箱子的数量
    float hranges[2];//值范围
    const float* ranges[1];//值范围的指针
    int channels[1];//要检查的通道数量

public:
    Histogram1D(){
        //准备一维直方图的默认参数
        histSize[0] = 256;//256个箱子
        hranges[0] = 0.0;//从0开始（包含0）
        hranges[1] = 256.0;//到256结束（不包含256）
        ranges[0] = hranges;//
        channels[0] = 0;//先关注通道0
    }

    Mat getHistogram(const Mat &image);

    Mat getHistogramImage(const Mat &image, int zoom = 1);//计算一维直方图，并返回它的图像

    static Mat getImageOfHistogram(const Mat &hist, int zoom);
};

#endif // HISTOGRAM1D_H
