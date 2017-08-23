#ifndef IMAGECOMPARATOR_H
#define IMAGECOMPARATOR_H

#include <colorhistogram.h>

class ImageComparator
{
private:
    Mat refH;// 基准直方图
    Mat inputH;// 输入图像的直方图

    ColorHistogram hist;// 生成直方图
    int nBins;// 每个颜色通道使用的箱子数量
public:
    ImageComparator():nBins(8){}

    // 设置比较直方图时使用的箱子数量
    void setNumberOfBins(int bins){
        nBins = bins;//为了得到更加可靠的相似度测量结果
    }

    // 计算基准图像的直方图
    void setRefrenceImage(const Mat &image){
        hist.setSize(nBins);
        refH = hist.getHistogram(image);
    }

    double compare(const Mat &image);
};

#endif // IMAGECOMPARATOR_H
