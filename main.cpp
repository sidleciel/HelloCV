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



#define WM_TAG "Original Image"

void flipSave(Mat &img){
    //flip，翻转
    Mat result;
    int flipCode = -1;//正数水平；0水平和垂直都翻转；负数垂直翻转
    flip(img, result, flipCode);
    imshow(WM_TAG, result);

    //保存
    string output = "output.png";//根据后缀保存格式
    imwrite(output, result);
}

void onMouse(int event, int x, int y, int flags, void* param){
    Mat *im = reinterpret_cast<Mat*>(param);

    switch (event)
    {
    case CV_EVENT_LBUTTONDOWN:
        cout << "at (" << x << "," << y << ") value is " << static_cast<int>(im->at<uchar>(Point(x, y))) << endl;
        break;
    default:
        break;
    }
}

int test1(){
    Mat img;
    cout << "This image is " << img.cols << "x" << img.rows << endl;
    if (img.empty()){
        cout << "This image is null." << endl << endl;
    }

    char ExePath[MAX_PATH];
    GetModuleFileName(NULL, (LPWSTR)ExePath, MAX_PATH);
    cout << ExePath << endl;

    //show iamge
    img = imread("1.png");
    cout << "This image is " << img.cols << "x" << img.rows << endl;
    if (img.empty()){
        cout << "img is empty" << endl;
        getchar();
        return 0;
    }
    namedWindow(WM_TAG);
    imshow(WM_TAG, img);

    //img = imread("1.png", CV_LOAD_IMAGE_GRAYSCALE);
    //imshow(WM_TAG, img);

    //img = imread("1.png", CV_LOAD_IMAGE_COLOR);
    //imshow(WM_TAG, img);

    cout << "This image has " << img.channels() << " channel(s)." << endl;

    setMouseCallback(WM_TAG, onMouse, &img);

    circle(img,
        Point(img.cols / 2, img.rows / 2),//圆心
        100,//半径
        Scalar(191),//颜色
        10);//描边
    imshow(WM_TAG, img);

    putText(img,
        "This is HaiMa.",
        Point(img.cols / 2 - 110, img.rows / 2 + 150),
        FONT_HERSHEY_PLAIN,//字体
        2.0,//字体大小
        255,
        2//厚度
        );
    imshow(WM_TAG, img);
    waitKey(0);
}


//测试函数，它创建一个图像
Mat funciton(){
    //创建图像
    Mat ima(500, 500, CV_8UC(1), 50);
    return ima;//返回图像
}

void test2(){
    string WM_T = "Image";
    string WM_T1 = "Image 1";
    string WM_T2 = "Image 2";
    string WM_T3 = "Image 3";
    string WM_T4 = "Image 4";
    string WM_T5 = "Image 5";
    //定义窗口
    namedWindow(WM_T1);
    namedWindow(WM_T2);
    namedWindow(WM_T3);
    namedWindow(WM_T4);
    namedWindow(WM_T5);
    namedWindow(WM_T);

    //创建一个240行X320列的图像
    Mat image1(240, 320, CV_8U, 100);
    imshow(WM_T, image1);//显示图像
    waitKey(0);//等待按键

    //重新分配一个新的图像
    image1.create(200, 200, CV_8U);
    image1 = 200;

    imshow(WM_T, image1);
    waitKey(0);

    //创建一个红色图像
    //通道次序依次为BGR
    Mat image2(240, 320, CV_8UC3, Scalar(0, 0, 255));
    Mat image5(240, 320, CV_8UC3, Scalar(0, 0, 255));
    //或者：
    //Mat image2(240, 320, CV_8UC3);
    //image2 = Scalar(0, 0, 255);

    imshow(WM_T2, image2);
    waitKey(0);

    //读入一个图像
    Mat image3 = imread("puppy.jpg");

    //所有这些图像都指向同一个数据块
    Mat image4(image3);
    image1 = image3;

    //这些是源图像的副本图像
    image3.copyTo(image2);
    image5 = image3.clone();

    //image1.release();
    //image4.release();

    //转换图像用来测试
    flip(image3, image3, 1);
    //flip(image2, image2, -1);
    //flip(image5, image5, 1);

    imshow(WM_T1, image1);
    imshow(WM_T2, image2);
    imshow(WM_T3, image3);
    imshow(WM_T4, image4);
    imshow(WM_T5, image5);
    waitKey(0);

    Mat gray = funciton();
    imshow(WM_T, gray);
    waitKey(0);

    image1 = imread("puppy.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    image1.convertTo(image2, CV_32F, 255 / 255.0, 0.0);

    imshow(WM_T, image2);
    waitKey(0);

}

void testROI(){
    Mat puppy = imread("puppy.jpg");
    Mat logo = imread("opencv-logo.png");

    Mat roi(puppy,
        Rect(puppy.cols - logo.cols, puppy.rows - logo.rows,
        logo.cols, logo.rows));
    logo.copyTo(roi);

    //方法2
    Mat imageRoi = puppy(Range(puppy.cols - logo.cols, logo.cols),
        Range(puppy.rows - logo.rows, logo.rows));
    logo.copyTo(roi);

    namedWindow("ROI");
    imshow("ROI", puppy);

    waitKey(0);
}

void testRoiMask(){
    Mat image = imread("puppy.jpg");
    Mat logo = imread("opencv-logo.png");

    // 在图像的右下角定义一个ROI
    Mat imageROI = image(cv::Rect(image.cols - logo.cols,
        image.rows - logo.rows,
        logo.cols, logo.rows));
    // 把标志作为掩码（必须是灰度图像）
    cv::Mat mask(logo);
    // 插入标志，只复制掩码不为0的位置
    logo.copyTo(imageROI, mask);

    namedWindow("ROI");
    imshow("ROI", image);

    waitKey(0);
}

void salt(Mat image, int n){

    int i, j;
    for (int k = 0; k < n; k++)
    {
        j = rand() % image.cols;
        i = rand() % image.rows;

        int salt = rand() % 255;

        if (image.type() == CV_8UC1)
        {
            image.at<uchar>(i, j) = salt;
        }
        else if (image.type() == CV_8UC3)
        {
            image.at<Vec3b>(i, j)[0] = salt;
            image.at<Vec3b>(i, j)[1] = salt;
            image.at<Vec3b>(i, j)[2] = salt;
        }
    }

}

void testSalt(){
    Mat img = imread("boldt.jpg");

    salt(img, 4000);

    namedWindow("Image");
    imshow("Image", img);

    waitKey(0);
}

void colorReduce(Mat image, int div = 64){
    int nl = image.rows;//行数
    int nc = image.cols*image.channels();//每行的像素数

    for (int j = 0; j < nl; j++)
    {
        //取得i行的地址
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++)
        {
            //处理每个像素
            //data[i] = data[i] / div*div + div / 2;//整数除法

            data[i] = data[i] - data[i] % div + div / 2;//取模算法
            //像素处理结束
        }//一行结束
    }
}

void colorReduce1(Mat image, int n = 4){
    int nl = image.rows;//行数
    int nc = image.cols*image.channels();//每行的像素数

    int div = pow(2, n);
    uchar mask = 0xFF << n;

    for (int j = 0; j < nl; j++)
    {
        //取得i行的地址
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++)
        {
            //处理每个像素
            *data &= mask;// 掩码
            *data++ += div >> 1;// 加上div/2
            //像素处理结束
        }//一行结束
    }
}

void colorReduce2(Mat &image, int div = 64)
{//连续图像
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


void testColorReduce(){
    Mat img = imread("boldt.jpg");

    colorReduce1(img, 6);

    namedWindow("Image");
    imshow("Image", img);

    waitKey(0);
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

void macMain()
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
    colorReduce2(src);
    imshow(winname, src);
    waitKey(0);

    destroyWindow(winname);
}


int main(int argc, char *argv[])
{
//    macMain();

    //test1();
    //test2();
    //testROI();
    //testRoiMask();
    //testSalt();
//    testColorReduce();


    return 0;
}