#include "contentfinder.h"



//使用全部通道[0,256]
Mat ContentFinder::find(const Mat &image)
{
    Mat result;

    hranges[0] = 0.0;
    hranges[1] = 256.0;

    channels[0]= 0; // 三个通道
    channels[1]= 1;
    channels[2]= 2;

    return find(image, hranges[0], hranges[1], channels);
}

//查找属于直方图的像素
Mat ContentFinder::find(const Mat &image, float minValue, float maxValue, int *channels)
{
    Mat result;
    hranges[0] = minValue;
    hranges[1] = maxValue;

    //直方图的维度数与通道列表一致
    for (int i = 0; i < histogram.dims; ++i) {
        this->channels[i] = channels[i];
    }

    calcBackProject(&image,
                    1,
                    channels,
                    histogram,
                    result,
                    ranges,
                    255.0);

    //对反向投影做阀值化，得到二值图像
    if (threshold>0.0) {
        cv::threshold(result, result, 255.0*threshold, 255.0, THRESH_BINARY);
    }

    return result;
}
