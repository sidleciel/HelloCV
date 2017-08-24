#include "morphofeature.h"

Mat MorphoFeature::getCorners(const Mat &image)
{
    //cv::MORPH_GRADIENT参数，即可实现此功能
    //边缘检测运算也叫Beucher梯度

    Mat result;

    //用十字元素膨胀
    dilate(image, result, cross);

    //用菱形元素腐蚀
    erode(result, result, diamond);

    Mat result2;
    //用X元素膨胀
    dilate(image, result2, x);

    //用正方形腐蚀
    erode(result2, result2, square);

    //比较两个经过闭合的图像，得到角点
    absdiff(result2, result, result);

    //应用阀值得到二值图像
    applyThreshold(result);

    return result;
}

void MorphoFeature::applyThreshold(Mat image)
{
    if (threshold > 0) {
        cv::threshold(image, image, threshold, 255, THRESH_BINARY);
    }
}

////用圈圈标记角点
//void drawOnImage(cv::Mat &image,
//                 const std::vector<cv::Point> &points,
//                 cv::Scalar color= cv::Scalar(255,255,255),
//                 int radius=3, int thickness=2) {
//    std::vector<cv::Point>::const_iterator it=points.begin();
//    while (it!=points.end()) {
//        // 角点处画圈
//        cv::circle(image,*it,radius,color,thickness);
//        ++it;
//    }
//}

void MorphoFeature::drawOnImage(const Mat &corners, Mat image)
{
    //    add(corners, image, image);

    Mat_<uchar>::const_iterator it = corners.begin<uchar>();
    Mat_<uchar>::const_iterator itend = corners.end<uchar>();

    int radius = 6, thickness = 1;
    Scalar color = Scalar(255, 255, 255);
    int i=0;
    while (it!=itend) {
        if (*it) {
            Point point = Point(i%image.cols, i/image.cols);
            // 角点处画圈
            cv::circle(image,point,radius,color,thickness);
        }
        ++it;
        i++;
    }
}
