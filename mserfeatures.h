#ifndef MSERFEATURES_H
#define MSERFEATURES_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace cv;

class MSERFeatures
{
private:
    cv::MSER mser; // MSER检测器
    double minAreaRatio; // 额外的排斥参数

    void getBouningRects(const Mat &image, std::vector<RotatedRect> &rects);
public:
    MSERFeatures(
            // 允许的尺寸范围
            int minArea=60, int maxArea=14400,
            // MSER面积/带边框矩形面积之比的最小值
            double minAreaRatio=0.5,
            // 用以测量稳定性的增量值
            int delta=5,
            // 最大允许面积变化量
            double maxVariation=0.25,
            // 子区域和父区域之间差距的最小值
            double minDiversity=0.2):
        mser(delta, minArea, maxArea, maxVariation, minDiversity),
        minAreaRatio(minAreaRatio) {}

    Mat getImageOfEllipse(const Mat &image, std::vector<RotatedRect> &rects, Scalar color = 255);
};

#endif // MSERFEATURE_H
