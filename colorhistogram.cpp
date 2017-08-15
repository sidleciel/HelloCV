#include "colorhistogram.h"

Mat ColorHistogram::getHistogram(const Mat &image)
{
    Mat hist;

    ranges[0] = 0.0;
    ranges[1] = 256.0;
    channels[0] = 0;
    channels[1] = 1;
    channels[2] = 2;

    calcHist(&image,
             1,//仅为一个图像的直方图
             channels,//使用的通道
             Mat(),//不使用掩码
             hist,//作为结果的直方图
             3,//这是三维的直方图
             histSize,//箱子数量
             ranges//像素值的范围
             );

    return hist;
}

SparseMat ColorHistogram::getSparseHistogram(const Mat &image)
{
    SparseMat hist(3,
                   histSize,
                   CV_32F);

    ranges[0] = 0.0;
    ranges[1] = 256.0;
    channels[0] = 0;
    channels[1] = 1;
    channels[2] = 2;

    calcHist(&image,
             1,//仅为一个图像的直方图
             channels,//使用的通道
             Mat(),//不使用掩码
             hist,//作为结果的直方图
             3,//这是三维的直方图
             histSize,//箱子数量
             ranges//像素值的范围
             );

    return hist;
}
