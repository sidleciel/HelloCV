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
    //颜色转换后的图像（色彩均匀感知）
    Mat converted;

    int getDistanceToTargetColor(const Vec3b& color) const;
    int getColorDistance(const Vec3b& color1,const  Vec3b& color2,const int method) const;
public:
    ColorDetector();
    ColorDetector(uchar blue,uchar green, uchar red, int maxDist);

    void setColorDistanceThreshold(int distance);
    int getColorDistanceThreshold() const;

    void setTargetColor(uchar blue,uchar green,uchar red);
    void setTargetColor(Vec3b color);
    Vec3b getTargetColor() const;

    Mat operator()(const Mat &image);

    Mat process(const Mat &image);
    Mat cvProcess(const Mat &image);

    //8位版本的色调在0-180之间，
    void dectectHScolor(const Mat &image,//输入图像
                        double minHue, double maxHue,//色调区间
                        double minSat, double maxSat,//饱和度区间
                        Mat &mask);//输出掩码
};

#endif // COLORDETECTOR_H
