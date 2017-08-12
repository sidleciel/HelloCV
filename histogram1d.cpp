#include "histogram1d.h"


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

static Mat getImageOfHistogram(const Mat &hist, int zoom)
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
