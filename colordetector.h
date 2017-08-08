#ifndef COLORDETECTOR_H
#define COLORDETECTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

class ColorDetector
{
private:
    //允许的最小差距
    int maxDist;
    //目标颜色
    Vec3b target;
    //存储二值映像的结果
    Mat result;

    int getDistanceToTargetColor(const Vec3b& color) const;
    int getColorDistance(const Vec3b& color1,const  Vec3b& color2,const int method) const;
public:
    ColorDetector();
    ColorDetector(uchar blue,uchar green, uchar red, int maxDist);

    Mat operator()(const Mat &image);

    Mat process(const Mat &image);
    Mat cvProcess(const Mat &image);

    void setColorDistanceThreshold(int distance);
    int getColorDistanceThreshold() const;

    void setTargetColor(uchar blue,uchar green,uchar red);
    void setTargetColor(Vec3b color);
    Vec3b getTargetColor() const;
};

#endif // COLORDETECTOR_H
