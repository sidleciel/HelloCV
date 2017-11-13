#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void cvSkinModel(Mat img, Mat &mask)
{
    if (img.empty())
        return;

    //椭圆皮肤模型
    Mat skinCrCbHist = Mat::zeros(Size(256, 256), CV_8UC1);
    ellipse(skinCrCbHist, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);


    Mat ycrcb_image;
    mask = Mat::zeros(img.size(), CV_8UC1);
    cvtColor(img, ycrcb_image, CV_BGR2YCrCb); //首先转换成到YCrCb空间

    for (int i = 0; i < img.cols; i++)   //利用椭圆皮肤模型进行皮肤检测
        for (int j = 0; j < img.rows; j++){
            Vec3b ycrcb = ycrcb_image.at<Vec3b>(j, i);
            if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0)
                mask.at<uchar>(j, i) = 255;
        }
    cv::erode(mask, mask, cv::Mat());
    cv::dilate(mask, mask, cv::Mat());
}


#ifdef CV_1
void cvThresholdOtsu(IplImage* src, IplImage* dst)
{
    int height=src->height,width=src->width,threshold=0;
    double histogram[256]={0};
    double average=0.0,max_variance=0.0,w=0.0,u=0.0;
    IplImage* temp=cvCreateImage(cvGetSize(src),src->depth,1);
    if(src->nChannels!=1)
        cvCvtColor(src,temp,CV_BGR2GRAY);
    else
        cvCopy(src,temp);

    unsigned char* p_temp=(unsigned char*)temp->imageData;

    //计算灰度直方图
    //
    for(int j=0;j<height;j++)
    {
        for(int i=0;i<width;i++)
        {
            histogram[p_temp[j*width+i]]++;
        }
    }
    for(int i=0;i<256;i++)
        histogram[i]=histogram[i]/(double)(height*width);

    //计算平局值
    for(int i=0;i<256;i++)
        average+=i*histogram[i];

    for(int i=0;i<256;i++)
    {
        w+=histogram[i];
        u+=i*histogram[i];

        double t=average*w-u;
        double variance=t*t/(w*(1-w));
        if(variance>max_variance)
        {
            max_variance=variance;
            threshold=i;
        }
    }

    cvThreshold(temp,dst,threshold,255,CV_THRESH_BINARY);

    cvReleaseImage(&temp);
}
#endif

void cvThresholdOtsu(Mat src, Mat &dst)
{
    if (src.empty()) return;
    //    a.将RGB图像转换到YCrCb颜色空间，提取Cr分量图像
    //    b.对Cr做自适应二值化处理（Ostu法）
    int height=src.rows, width=src.cols, threshold=0;
    double histogram[256]={0};
    double average=0.0,max_variance=0.0,w=0.0,u=0.0;

    Mat temp(src.size(), src.type());
    if (src.channels() != 1)
    {
        cv::cvtColor(src, temp, CV_BGR2GRAY);
    } else {
        src.copyTo(temp);
    }

    unsigned char* p_temp=(unsigned char*)temp.data;

    //计算灰度直方图
    for(int j=0;j<height;j++)
    {
        for(int i=0;i<width;i++)
        {
            histogram[p_temp[j*width+i]]++;
        }
    }
    for(int i=0;i<256;i++)
        histogram[i]=histogram[i]/(double)(height*width);

    //计算平局值
    for(int i=0;i<256;i++)
        average+=i*histogram[i];

    for(int i=0;i<256;i++)
    {
        w+=histogram[i];
        u+=i*histogram[i];

        double t=average*w-u;
        double variance=t*t/(w*(1-w));
        if(variance>max_variance)
        {
            max_variance=variance;
            threshold=i;
        }
    }

    cv::threshold(temp,dst,threshold,255,CV_THRESH_BINARY);
}
