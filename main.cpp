//#include "mainwindow.h"
//#include <QApplication>

//int main(int argc, char *argv[])
//{
//    QApplication a(argc, argv);
//    MainWindow w;
//    w.show();

//    return a.exec();
//}



#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#define RES "/Users/xietao/Downloads/qtProjects/helloCv/"

void colorReduce(Mat &image, int div = 64)
{
    int nl = image.rows;//行数
    int nc = image.cols*image.channels();

    if(image.isContinuous())
    {
        cout<<"This image is continous."<<endl;
        nc *= nl;
        nl = 1;//成了一长的一维数组

        //没有填充数据
        image.reshape(1,1);//新的通道数，新的行数
    }

    for(int j = 0; j< nl ;j++)
    {
        uchar *data = image.ptr<uchar>(j);

        for(int i=0;i<nc ;i++)
        {
            data[i] = data[i]/div*div+div/2;
        }
    }
}

void* ptrTest(Mat &image)
{//此方法不推荐使用
    uchar* data = image.data;//获取图像的首地址

    //image.step 获取一行像素的总字节数（包括填充像素）
    data += image.step;//下一行的地址

    //(j,i)的像素的地址，即image.at(j,i)
    int j,i;
    data = image.data + image.step * j + image.elemSize()*i;
}


int main(int argc, char *argv[])
{
    Mat src = imread(RES "logo.jpg");
    if(src.empty())
    {
        cerr << "Please check the path of input image!" << endl;
        return -1;
    }
    const string winname = "src";
    namedWindow(winname, WINDOW_AUTOSIZE);
    imshow(winname, src);
    waitKey(0);

    src = imread(RES "boldt.jpg");
    colorReduce(src);
    imshow(winname, src);
    waitKey(0);


    destroyWindow(winname);
    return 0;
}
