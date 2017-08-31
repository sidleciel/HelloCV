#include "mserfeatures.h"


void MSERFeatures::getBouningRects(const Mat &image, std::vector<RotatedRect> &rects)
{
    std::vector<std::vector<Point>> points;
    mser(image, points);

    // 针对每个检测到的特征
    for (std::vector<std::vector<Point>>::iterator it = points.begin();
         it != points.end(); ++it) {
        // 提取带边框的矩形
        RotatedRect rr = minAreaRect(*it);

        // 检查面积比例
        if (it->size() > minAreaRatio * rr.size.area()) {
            rects.push_back(rr);
        }
    }
}

// 画出对应每个MSER的旋转椭圆
Mat MSERFeatures::getImageOfEllipse(const Mat &image, std::vector<RotatedRect> &rects, Scalar color)
{
    // 画到这个图像上
    Mat output = image.clone();
    // 得到MSER特征
    getBouningRects(image, rects);
    // 针对每个检测到的特征
    for (std::vector<RotatedRect>::iterator it = rects.begin();
         it != rects.end(); ++it) {
        cv::ellipse(output, *it, color);
    }

    return output;
}
