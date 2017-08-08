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

Mat ColorDetector::process(const Mat &image)
{
    //必要时重新分配二值映像
    //    与输入图像相同，不过是单通道
    result.create(image.size(),CV_8U);

    //循环处理

    //取得迭代器
    Mat_<Vec3b>::const_iterator it = image.begin<Vec3b>();
    Mat_<Vec3b>::const_iterator itend = image.end<Vec3b>();

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
    //次序为BGR
    target = Vec3b(blue,green,red);
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
