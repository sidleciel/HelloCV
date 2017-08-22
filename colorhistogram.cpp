#include "colorhistogram.h"

Mat ColorHistogram::getHistogram(const Mat &image)
{
    Mat hist;

    hranges[0] = 0.0;
    hranges[1] = 256.0;
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

// 计算一维色调直方图（带掩码）
// BGR的原图转换成HSV
// 忽略低饱和度的像素
Mat ColorHistogram::getHueHistogram(const Mat &image, int minSaturation)
{
    Mat hist;

    // 转换成HSV色彩空间
    Mat hsv;
    cvtColor(image, hsv, CV_BGR2HSV);

    // 掩码（可用或可不用）
    Mat mask;

    if (minSaturation>0) {
        // 把3个通道分割进3个图像
        std::vector<Mat> v;
        split(hsv, v);

        // 屏蔽低饱和度的像素
        threshold(v[1], mask, minSaturation, 255.0, THRESH_BINARY);
    }

    // 准备一维色调直方图的参数
    hranges[0] = 0.0;
    hranges[1] = 180.0;// 范围为0~180
    channels[0] = 0;// 色调通道

    // 计算直方图
    calcHist(&hsv,
             1,// 只有一个图像的直方图
             channels,// 用到的通道
             mask,// 二值掩码
             hist,
             1,// 这是一维直方图
             histSize,// 箱子数量
             ranges// 像素值范围
             );

    return hist;
}

SparseMat ColorHistogram::getSparseHistogram(const Mat &image)
{
    SparseMat hist(3,
                   histSize,
                   CV_32F);

    hranges[0] = 0.0;
    hranges[1] = 256.0;
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
