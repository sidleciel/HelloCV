#include "imagecomparator.h"


double ImageComparator::compare(const Mat &image)
{
    inputH = hist.getHistogram(image);
    //    CV_COMP_INTERSECT 逐个箱子地比较每个直方图中的数值，并保存最小的值。然后把这些最小值累加，作为相
    //    似度测量值。因此，两个没有相同颜色的直方图得到交叉值为0，而两个完全相同的直方图得到的值就等于像素总数。

    //    卡方测量法（CV_COMP_CHISQR标志），累加各箱子的归一化平方差；
    //    关联性算法（CV_COMP_CORREL标志），基于信号处理中的归一化交叉关联操作符测量两个信号的相似度；
    //    Bhattacharyya测量法（CV_COMP_BHATTACHARYYA标志），用在统计学中，评估两个概率分布的相似度。
    return cv::compareHist(inputH, refH, CV_COMP_INTERSECT);
}
