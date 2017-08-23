#ifndef INTEGRALIMAGE_H
#define INTEGRALIMAGE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

template <typename T, int N>
class IntegralImage
{
    Mat integralImage;
public:
    IntegralImage(Mat image){
        //(很耗时)计算积分图像
        cv::integral(image, integralImage, DataType<T>::type);
    }

    //通过访问4个像素，计算任何尺寸子区域的累计值
    cv::Vec<T,N> operator()(int xo, int yo,
                            int width, int height) {
        //（xo, yo）处的窗口，尺寸为width × height
        return (integralImage.at<cv::Vec<T,N>>(yo+height,xo+width)
                -integralImage.at<cv::Vec<T,N>>(yo+height,xo)
                -integralImage.at<cv::Vec<T,N>>(yo,xo+width)
                +integralImage.at<cv::Vec<T,N>>(yo,xo));
    }
};

#endif // INTEGRALIMAGE_H
