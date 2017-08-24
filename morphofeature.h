#ifndef MORPHOFEATURE_H
#define MORPHOFEATURE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

class MorphoFeature
{
private:
    //用于产生二值图像的阀值
    int threshold;

    //用于检测角点的结构元素
    Mat_<uchar> cross;
    Mat_<uchar> diamond;
    Mat_<uchar> square;
    Mat_<uchar> x;

    void applyThreshold(Mat image);

public:
    MorphoFeature():threshold(-1),
        cross(5,5), diamond(5,5), square(5,5), x(5,5){
        // 创建十字形结构元素
        cross <<
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                1, 1, 1, 1, 1,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0;

        // 用类似方法创建其他结构元素
        diamond <<
                0, 0, 1, 0, 0,
                0, 1, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 1, 1, 0,
                0, 0, 1, 0, 0;
        square <<
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1;
        x <<
                1, 0, 0, 0, 1,
                0, 1, 0, 1, 0,
                0, 0, 1, 0, 0,
                0, 1, 0, 1, 0,
                1, 0, 0, 0, 1;
    }

    void setThreshold(int threshold){
        this->threshold = threshold;
    }

    Mat getCorners(const Mat &image);
    void drawOnImage(const Mat &corners, Mat image);
};

#endif // MORPHOFEATURE_H
