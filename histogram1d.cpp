#include "histogram1d.h"

Mat Histogram1D::getImageOfHistogram(const Mat &hist, int zoom)
{
    double minVal = 0;
    double maxVal = 0;
    cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

    int histSize = hist.rows;

    Mat histImg(histSize*zoom, histSize*zoom, CV_8U, Scalar(255));

    //设置最高点为90%（即图像的高度）的箱子个数
    int hpt = static_cast<int>(0.9*histSize);

    for (int h = 0; h < histSize; ++h) {
        float binVal = hist.at<float>(h);

        if (binVal>0) {
            int intensity = static_cast<int>(binVal * hpt / histSize);
            cv::line(histImg,
                     cv::Point(h*zoom, histSize*zoom),
                     cv::Point(h*zoom, (histSize-intensity)*zoom),
                     cv::Scalar(0),
                     zoom);
        }


    }

    return histImg;
}

Mat Histogram1D::getHistogram(const Mat &image)
{
    Mat hist;

    calcHist(&image,
             1,//仅为一个图像的直方图
             channels,//使用的通道
             Mat(),//不使用掩码
             hist,//作为结果的直方图
             1,//这是一维的直方图
             histSize,//箱子数量
             ranges//像素值的范围
             );

    return hist;
}

Mat Histogram1D::getHistogramImage(const Mat &image, int zoom)
{
    Mat hist = getHistogram(image);

    return getImageOfHistogram(hist, zoom);
}

Mat Histogram1D::applyLookUp(const Mat &image,//输入图像
                             const Mat &lookup)//uchar 类型的1x256数组
{
    Mat result;

    //应用查找表
    LUT(image,lookup,result);
    return result;
}

cv::Mat Histogram1D::stretch(const cv::Mat &image, int minValue) {
    // 首先计算直方图
    cv::Mat hist= getHistogram(image);
    // 找到直方图的左边限值
    int imin= 0;
    for( ; imin < histSize[0]; imin++ ) {
        // 忽略数量少于minValue项目的箱子
        if (hist.at<float>(imin) > minValue)
            break;
    }
    // 找到直方图的右边限值
    int imax= histSize[0]-1;
    for( ; imax >= 0; imax-- ) {
        // 忽略数量少于minValue的箱子
        if (hist.at<float>(imax) > minValue)
            break;
    }
    // 创建查找表
    int dim(256);
    cv::Mat lookup(1, // 一维
                   &dim, // 256个项目
                   CV_8U); // uchar类型
    // 构建查找表
    for (int i=0; i<256; i++) {
        // 在imin和imax之间伸展
        if (i < imin) lookup.at<uchar>(i)= 0;
        else if (i > imax) lookup.at<uchar>(i)= 255;
        // 线性映射
        else lookup.at<uchar>(i)=
                cvRound(255.0*(i-imin)/(imax-imin));
    }
    // 应用查找表
    cv::Mat result;
    result= applyLookUp(image,lookup);
    return result;
}
