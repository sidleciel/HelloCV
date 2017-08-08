#include "colordetector.h"

ColorDetector::ColorDetector():maxDist(100),target(0,0,0)
{

}

ColorDetector::ColorDetector(uchar blue,uchar green, uchar red, int maxDist = 100):maxDist(maxDist)
{
    setTargetColor(blue,green,red);
}

Mat ColorDetector::operator() (const Mat &image)
{
    return process(image);
}

void ColorDetector::dectectHScolor(const Mat &image, double minHue, double maxHue, double minSat, double maxSat, Mat &mask)
{
    //转换到HSV空间
    Mat hsv;

    cvtColor(image,hsv,CV_BGR2HSV);

    //分割3个通道，到3个图像
    vector<Mat> channels;
    split(hsv,channels);
    //channels[0] 是色调
    //channels[1] 是饱和度
    //channels[2] 是亮度

    //色调掩码
    Mat mask1;
    threshold(channels[0],mask1,maxHue,255,THRESH_BINARY_INV);//小于
    Mat mask2;
    threshold(channels[0],mask2,minHue,255,THRESH_BINARY);//大于

    Mat hueMask;
    if (minHue < maxHue) {
        hueMask = mask1 & mask2;
    } else {//如果区间穿越0度中轴线
        hueMask = mask1 | mask2;
    }

    //饱和度掩码
    threshold(channels[1],mask1,maxSat,255,THRESH_BINARY_INV);//小于
    threshold(channels[1],mask2,minSat,255,THRESH_BINARY);//大于

    Mat satMask;
    satMask = mask1 & mask2;

    //组合掩码
    mask = hueMask & satMask;
}

Mat ColorDetector::process(const Mat &image)
{
    //必要时重新分配二值映像
    //    与输入图像相同，不过是单通道
    result.create(image.size(),CV_8U);

    cvtColor(image,converted,CV_BGR2Lab);

    //循环处理

    //取得迭代器
    Mat_<Vec3b>::const_iterator it = converted.begin<Vec3b>();
    Mat_<Vec3b>::const_iterator itend = converted.end<Vec3b>();

    Mat_<uchar>::iterator itout = result.begin<uchar>();

    //对比每个像素
    for (; it != itend; ++it,++itout) {
        //比较与目标颜色的差距
        if (getDistanceToTargetColor(*it)<=maxDist) {
            *itout = 255;
        }else{
            *itout = 0;
        }
    }

    return result;
}

Mat ColorDetector::cvProcess(const Mat &image)
{
    Mat output;
    //计算与目标图像距离的绝对值
    absdiff(image,Scalar(target),output);

    //把通道分割进3个图像
    vector<Mat> images;
    split(output,images);

    //3个通道想加（有可能出现饱和的情况）
    output = images[0] + images[1] + images[2];

    //应用阀值
    threshold(output,//输入图像
              output,//输出图像
              maxDist,//阀值（必须<256）
              255,//最大值
              THRESH_BINARY_INV);//阀值化模式

    return output;
}

//计算与目标颜色的差距
int ColorDetector::getDistanceToTargetColor(const Vec3b &color) const
{
    return getColorDistance(color,target,1);
}


int ColorDetector::getColorDistance(const Vec3b &color1,const  Vec3b &color2,const int method = 0) const
{
    if (method==0) {
        //计算两个颜色之间的城区距离
        return abs(color1[0]-color2[0])+
                abs(color1[1]-color2[1])+
                abs(color1[2]-color2[2]);
    } else if (method==1) {
        //计算向量的欧几里德范数的函数
        return static_cast<int>(
                    norm<int,3>(Vec3i(color1[0]-color2[0], color1[1]-color2[1], color1[2]-color2[2])));
    } else if (method == 2) {
        Vec3b dist;
        absdiff(color1,color2,dist);
        return sum(dist)[0];
    }

    return 0;
}

//设置颜色差距的阀值
//阀值必须为正数，否则为0
void ColorDetector::setColorDistanceThreshold(int distance)
{
    if (distance<0) {
        distance = 0;
    }
    maxDist = distance;
}

//取得颜色差距的阀值
int ColorDetector::getColorDistanceThreshold() const
{
    return maxDist;
}

//设置需要检测的颜色
void ColorDetector::setTargetColor(uchar blue, uchar green, uchar red)
{
    //临时的单像素图像
    Mat tmp(1,1,CV_8UC3);
    tmp.at<Vec3b>(0,0) = Vec3b(blue,green,red);

    //目标像素转换为Lab色彩空间
    cvtColor(tmp,tmp,CV_BGR2Lab);

    target = tmp.at<Vec3b>(0,0);

    //次序为BGR
    //    target = Vec3b(blue,green,red);
}

//设置需要检测的颜色
void ColorDetector::setTargetColor(Vec3b color)
{
    target = color;
}

//取得需要检测的颜色
Vec3b ColorDetector::getTargetColor() const
{
    return target;
}
