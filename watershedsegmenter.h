#ifndef WATERSHEDSEGMENTER_H
#define WATERSHEDSEGMENTER_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;


class WatershedSegmenter
{
private:
    Mat markers;
public:
    WatershedSegmenter();

    void setMarkers(const Mat &markerImage){
        // 转换成整数型图像
        markerImage.convertTo(markers, CV_32S);
    }

    // 以图像形式返回结果
    cv::Mat getSegmentation() {
        cv::Mat tmp;
        // 从32S到8U（0-255）会进行饱和运算，所以像素高于255的一律复制为255
        markers.convertTo(tmp,CV_8U);//
        return tmp;
    }

    // 以图像形式返回分水岭（我理解的是分割线）
    cv::Mat getWatersheds() {
        cv::Mat tmp;
        //在设置标记图像，即执行setMarkers（）后，边缘的像素会被赋值为-1，其他的用正整数表示
        //下面的这个转换可以让边缘像素变为-1*255+255=0，即黑色，其余的溢出，赋值为255，即白色。
        markers.convertTo(tmp, CV_8U, 255, 255);
        return tmp;
    }

    Mat process(const Mat &image){
        //Only 32-bit, 1-channel output images are supported
        cv::watershed(image, markers);
        return markers;
    }
};

#endif // WATERSHEDSEGMENTER_H
