#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace cv;

#include <stdafx.h>

#include <colordetector.h>
#include <histogram1d.h>
#include <colorhistogram.h>
#include <contentfinder.h>
#include <imagecomparator.h>
#include <integralimage.h>
#include <morphofeature.h>
#include <watershedsegmenter.h>
#include <mserfeatures.h>

#ifdef SHOW_WIN_FORM
#include "mainwindow.h"
#include <QApplication>
#endif

// 方法计时
int64 tickCont = 0;

int64 startCount()
{
    return tickCont = cv::getTickCount();
}

int64 getTimeCount()
{
    int64 ret = (cv::getTickCount() - tickCont)/cv::getTickFrequency();
    tickCont = cv::getTickCount();
    return ret;
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

void showImage(Mat &image, String tag = WM_TAG, int wait = 0)
{
    namedWindow(tag);
    imshow(tag, image);
    setMouseCallback(tag, onMouse, &image);

    if(wait==0)
        waitKey(0);
}

void showImage(Mat &image, int wait)
{
    showImage(image, WM_TAG, wait);
}

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

int test1(){
    Mat img;
    cout << "This image is " << img.cols << "x" << img.rows << endl;
    if (img.empty()){
        cout << "This image is null." << endl << endl;
    }
    
#ifdef Q_OS_WIN
    char ExePath[MAX_PATH];
    GetModuleFileName(NULL, (LPWSTR)ExePath, MAX_PATH);
    cout << ExePath << endl;
#endif
    
    //show iamge
    img = imread("boldt.jpg");
    cout << "This image is " << img.cols << "x" << img.rows << endl;
    if (img.empty()){
        cout << "img is empty" << endl;
        getchar();
        return 0;
    }
    namedWindow(WM_TAG);
    imshow(WM_TAG, img);
    
    //img = imread("boldt.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    //imshow(WM_TAG, img);
    
    //img = imread("boldt.jpg", CV_LOAD_IMAGE_COLOR);
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
    Mat image3 = imread("puppy.bmp");
    
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
    
    image1 = imread("puppy.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    image1.convertTo(image2, CV_32F, 255 / 255.0, 0.0);
    
    imshow(WM_T, image2);
    waitKey(0);
    
}

void testROI(){
    Mat puppy = imread("puppy.bmp");
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
    Mat image = imread("puppy.bmp");
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
    Mat img = imread(RES "boldt.jpg");
    
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
{//对连续图像的高效扫描
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

void colorReduce3(Mat &image, int div = 64)
{//用迭代器扫描图像
    cout << "用迭代器扫描图像" << endl;
    //    常量迭代器
    //    cv::MatConstIterator_<cv::Vec3b> it;
    //    或者
    //    cv::Mat_<cv::Vec3b>::const_iterator it;
    
    //获取初始位置迭代器
    Mat_<Vec3b>::iterator it = image.begin<Vec3b>();
    //获取结束位置迭代器
    Mat_<Vec3b>::iterator itend = image.end<Vec3b>();
    
    //    方法2：不需要制定返回迭代器类型
    Mat_<Vec3b> cimage(image);
    it = cimage.begin();
    itend = cimage.end();
    
    //循环遍历所有像素
    for(;it!=itend;++it)
    {
        (*it)[0] = (*it)[0]/div*div + div/2;
        (*it)[1] = (*it)[1]/div*div + div/2;
        (*it)[2] = (*it)[2]/div*div + div/2;
    }
}

void colorReduce4(Mat &image, int n=4)
{
    uchar mask = 0xFF<<n;
    int div = pow(2,n);
    image = (image&Scalar(mask,mask,mask))+Scalar(div/2,div/2,div/2);
}

void colorReduce5(Mat &image, int div = 64)
{//查找表方法,增强图像对比度

    Mat lookup(1,256, CV_8U);

    for (int i = 0; i < 256; ++i) {
        lookup.at<uchar>(i) = i/div*div+div/2;
    }

    LUT(image,lookup,image);
}

void testColorReduce(int method = 0){
    Mat img = imread(RES "boldt.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    
    if(img.empty())
    {
        cout<<(RES "boldt.jpg") << endl <<"This image is empty."<<endl;
        return;
    }
    
    const int64 start = getTickCount();
    if (method==0) {
        colorReduce(img);
    }else if (method==1) {
        colorReduce1(img);
    }else if (method==2) {
        colorReduce2(img);
    }else if (method==3) {
        colorReduce3(img);
    }else if (method==4) {
        colorReduce4(img);
    }else if (method==5) {
        colorReduce5(img);
    }

    double duration = (getTickCount()-start)/getTickFrequency();
    cout<<"colorReduce duration="<<duration<<endl;
    
    
    namedWindow("Image");
    imshow("Image", img);
    
    waitKey(0);
}

void* ptrTest(Mat &image)
{//低层次指针算法，此方法不推荐使用
    uchar* data = image.data;//获取图像的首地址
    
    //image.step 获取一行像素的总字节数（包括填充像素）
    data += image.step;//下一行的地址
    
    //(j,i)的像素的地址，即image.at(j,i)
    int j,i;
    data = image.data + image.step * j + image.elemSize()*i;
}

void sharpen(const Mat &image, Mat &result)
{
    result.create(image.size(),image.type());
    int nchannels = image.channels();
    
    //    计算锐化的数值：
    //            sharpened_pixel= 5*current-left-right-up-down;
    
    for(int j=1;j<image.rows-1;j++){
        const uchar* previous = image.ptr<const uchar>(j-1);
        const uchar* current = image.ptr<const uchar>(j);
        const uchar* next = image.ptr<const uchar>(j+1);
        
        uchar* output = result.ptr<uchar>(j);
        
        for(int i = nchannels;i<image.cols-1;i++)
        {
            *output++ = saturate_cast<uchar>(5*current[i] - current[i-nchannels] - current[i+nchannels]
                    - previous[i]
                    - next[i]);
        }
        
        result.row(0).setTo(Scalar(0));
        result.row(image.rows-1).setTo(Scalar(0));
        result.col(0).setTo(Scalar(0));
        result.col(image.cols-1).setTo(Scalar(0));
    }
}

int osxHello()
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

void shapern2D(const Mat &image, Mat &result)
{
    //构造内核，所有入口都初始化为0
    Mat kernel(3,3,CV_32F,Scalar(0));
    //对内核赋值
    kernel.at<float>(1,1) = 5.0;
    kernel.at<float>(0,1) = -1.0;
    kernel.at<float>(1,0) = -1.0;
    kernel.at<float>(1,2) = -1.0;
    kernel.at<float>(2,1) = -1.0;

    //对图像滤波
    filter2D(image,result,image.depth(),kernel);

    //大内核filter2D是特别有效的，因为使用了更高效的算法

}


void testSharpen(int method = 0){
    Mat img = imread(RES "boldt.jpg");

    if(img.empty())
    {
        cout<<(RES "boldt.jpg") << endl <<"This image is empty."<<endl;
        return;
    }

    const int64 start = getTickCount();
    Mat result;
    if (method == 0) {
        cvtColor(img,img,CV_BGR2GRAY);
        sharpen(img, result);
    }else if(method == 1){
        shapern2D(img,result);
    }else{
        result = img;
    }
    double duration = (getTickCount()-start)/getTickFrequency();
    cout<<"sharpen duration="<<duration<<endl;


    namedWindow("Image");
    imshow("Image", result);

    waitKey(0);
}

void testImgCalc(int method = 0)
{

    Mat resultC;
    Mat imageA = imread(RES "boldt.jpg");
    Mat imageB = imread(RES "rain.jpg");

    double k1,k2,k3,k;

    if (method==0) {
        k1 = k2 = 0.7;
        k3 = 0.;
        //cv[i] = a[I]*k1 +b[I]*k2+k3;
        addWeighted(imageA,k1,imageB,k2,k3,resultC);
    }else if (method ==1) {
        //cv[i] = a[i] +b[I];
        add(imageA,imageB,resultC);
    }else if (method==2) {
        k=255;
        //cv[i] = a[i] +k;
        add(imageA,Scalar(k),resultC);
    }else if (method==3) {
        k=0.7;
        //cv[i] = a[i] *k+b[I];
        scaleAdd(imageA,k,imageB,resultC);
    }else if (method==4) {
        Mat mask;
        cvtColor(imageA,mask,CV_BGR2GRAY);

        //if(mask[i])c[i] = a[i]+b[i]
        add(imageA,imageB,resultC,mask);
    }

    namedWindow("Image");
    imshow("Image", resultC);
    waitKey(0);
}

void testSplit()
{

    Mat image = imread(RES "boldt.jpg");
    Mat imageB = imread(RES "rain.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    Mat result;

    //创建3个相同的向量
    vector<Mat> planes;
    //分割一个3通道图像到一个单通道图像
    split(image,planes);
    //加到蓝色通道上
    planes[0]+=imageB;
    //合并三个单通道图像为一个3通道图像
    merge(planes,result);

    namedWindow("Image");
    imshow("Image", result);
    waitKey(0);
}

//重映射图像，创建波浪形效果
void wave(const Mat &image, Mat &result)
{
    if(image.empty())
    {
        return;
    }

    //映射参数
    Mat srcX(image.rows,image.cols,CV_32F);
    Mat srcY(image.rows,image.cols,CV_32F);

    //创建映射参数
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            //i，j像素的新位置
            srcX.at<float>(i,j) = j;//保持在同一列
            //原来在第i行的像素，现在根据一个正弦行数来移动
            srcY.at<float>(i,j) = i+5*sin(j/10.0);

            //            反转测试
            //            srcX.at<float>(i,j) = image.cols-j;
            //            srcY.at<float>(i,j) = image.rows-i;
        }
    }

    //应用映射函数
    remap(image,result,srcX,srcY,INTER_LINEAR);//插值法
}

void testRemap()
{
    Mat image = imread(RES "boldt.jpg");
    Mat result;

    wave(image,result);

    namedWindow("Image");
    if(!result.empty())
    {
        imshow("Image", result);
    }
    waitKey(0);
}

void testColorDetector(int method = 0)
{
    //1.创建图像处理对象
    ColorDetector cdetector;
    //2.读取输入的图像
    Mat img = imread(RES "boldt.jpg");
    if (img.empty()) {
        return;
    }


    if (method ==1 ) {//用仿函数处理
        cout<<"用仿函数处理..."<<endl;
        ColorDetector colordetector(230,190,130,100);
        Mat result = colordetector(img);

        if (result.empty()) {
            return;
        }
        namedWindow("Image");
        imshow("Image",result);
        waitKey(0);
        return;
    }

    //3.设置输入参数
    Vec3b color(230,190,130);
    //    Vec3b color(255,255,255);
    cdetector.setTargetColor(color);//天蓝色
    //4.处理图像并显示结果
    namedWindow("Image");
    cout<<"processing..."<<endl;
    imshow("Image",cdetector.cvProcess(img));
    cout<<"process finished."<<endl;
    waitKey(0);
}

void  testHSV()
{
    Mat image = imread(RES "boldt.jpg");
    Mat hsv;

    //转换为HSV色彩空间
    cvtColor(image,hsv,CV_BGR2HSV);

    vector<Mat> channels;
    split(hsv,channels);

    //所有颜色的亮度通道变为255
    channels[2] = 255;
    //重新合并通道
    merge(channels,hsv);

    //转回BGR
    Mat result;
    cvtColor(hsv,result,CV_HSV2BGR);

    namedWindow(WM_TAG);
    imshow(WM_TAG,result);

    waitKey(0);
}

void detectHScolor(const Mat &image, double minHue, double maxHue, double minSat, double maxSat, Mat &mask)
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

void testDetectHSV()
{
    Mat image = imread(RES "girl.jpg");

    Mat mask;
    detectHScolor(image,
                  160,10,//色调从320度到20度
                  25,166,//饱和度从0.1到0.65
                  mask);

    Mat detected(image.size(),CV_8UC3,Scalar(0,0,0));

    image.copyTo(detected,mask);

    showImage(detected);
}

void testHistogram1D()
{
    Mat image = imread(RES "group.jpg", CV_LOAD_IMAGE_GRAYSCALE);//以黑白方式打开

    Histogram1D h;

    Mat hist = h.getHistogram(image);

    //    //循环遍历每个箱子
    //    for (int i = 0; i < 256; ++i) {
    //        cout << "Value " << i << " = " << hist.at<float>(i) <<endl;
    //    }

    Mat result = h.getHistogramImage(image);

    namedWindow("Image");
    imshow("Image", image);

    if(!result.empty()){
        namedWindow(WM_TAG);
        imshow(WM_TAG, result);

        setMouseCallback(WM_TAG, onMouse, &result);


        Mat thresholded;
        threshold(image,thresholded,125,255,THRESH_BINARY);
        namedWindow("Threshold");
        imshow("Threshold", thresholded);

        waitKey(0);
    }
}

void testLut()
{
    int dim(256);
    Mat lut(1,//一维
            &dim,//256个项目
            CV_8U);//uchar类型

    for (int i = 0; i < dim; ++i) {
        lut.at<uchar>(i) = 255 -i;
    }

    Mat image = imread(RES "boldt.jpg", CV_LOAD_IMAGE_GRAYSCALE);//以黑白方式打开
    Mat result = Histogram1D::applyLookUp(image, lut);

    showImage(result);
}

void testStrech()
{
    Mat image = imread(RES "group.jpg", CV_LOAD_IMAGE_GRAYSCALE);//以黑白方式打开
    showImage(image);

    Histogram1D h;

    // 把1%的像素设为黑色，1%的设为白色
    cv::Mat streteched = h.stretch(image,0.01f);

    showImage(streteched);
}

void testBackProject()
{
    Mat image = imread(RES "beach.jpg", CV_LOAD_IMAGE_GRAYSCALE);//以黑白方式打开

    namedWindow("WM_TAG");
    imshow("WM_TAG", image);
    setMouseCallback("WM_TAG", onMouse, &image);
    waitKey(0);

    //
    Mat imageRoi;
    imageRoi = image(Rect(410,180,20,30));

    Histogram1D h;
    Mat hist = h.getHistogram(imageRoi);

    normalize(hist,hist,1.0);

    int channels[1];
    float hranges[2];
    const float* ranges[1];
    Mat result;

    channels[0] = 0;
    hranges[0] = 1.0f;
    hranges[1] = 256.0f;
    ranges[0] = hranges;

    calcBackProject(&image,
                    1,//一个图像
                    channels,//用到的通道，取决于图像的维度
                    hist,//需要反向投影的直方图
                    result,//反向投影得到的结果
                    ranges,//值的范围
                    255.0);//选用的换算系数，把概率值从1映射到255

    double thresh = 60.0;
    threshold(result, result, thresh,255, THRESH_BINARY);

    imshow(WM_TAG, result);

    waitKey(0);
}

void testBackProject1()
{
    Mat image = imread(RES "beach.jpg");//以黑白方式打开
    Rect rect(130,30,20,30);
    rectangle(image, rect, Scalar(0,0,255));

    namedWindow("WM_TAG");
    imshow("WM_TAG", image);
    setMouseCallback("WM_TAG", onMouse, &image);
    waitKey(0);

    // 提取兴趣区域
    Mat imageRoi;
    imageRoi = image(rect);

    ColorHistogram hc;
    // 取得3D颜色直方图（每个通道含8个箱子）
    hc.setSize(8); // 8 × 8 × 8

    Mat chist = hc.getHistogram(imageRoi);

    ContentFinder finder;
    finder.setHistogram(chist);
    finder.setThreshold(0.25f);

    Mat result = finder.find(image);
    showImage(result);
}

void testHsvBackProject()
{
    //读取参考图像
    Mat image = imread(RES "baboon1.jpg");

    //狒狒脸部ROI
    Mat imageROI;
    Rect rect(110,260,35,40);
    imageROI = image(rect);
    if (imageROI.empty()) {
        return;
    }

    rectangle(image, rect, Scalar(0,0,255), 1, 1, 0);
    namedWindow(WM_TAG);
    imshow(WM_TAG, image);
    setMouseCallback(WM_TAG, onMouse, &image);
    waitKey(0);

    int minSat = 65;
    ColorHistogram hc;
    Mat colorHist = hc.getHueHistogram(imageROI, minSat);

    ContentFinder finder;
    finder.setHistogram(colorHist);

    // 转换成HSV色彩空间
    image = imread(RES "baboon3.jpg");
    Mat hsv;
    cvtColor(image, hsv, CV_BGR2HSV);
    // 得到色调直方图的反向投影
    int ch[1] = {0};
    finder.setThreshold(-0.1f);// 不做阈值化
    Mat result = finder.find(hsv, 0.0f, 180.0f, ch);

    //初始搜索位置
    rectangle(image, rect, Scalar(0,0,255), 1, 1, 0);

    TermCriteria criteria(TermCriteria::MAX_ITER,
                          10,
                          0.01);//迭代停止条件为：最大迭代10次，中心偏移距离小于0.01
//    #include <opencv2/video/tracking.hpp>//meanShift 所在头文件
    cout << "meanShift=" << cv::meanShift(result, rect, criteria) << endl;
    //搜索结束位置
    rectangle(image, rect, Scalar(0,255,0), 1, 1, 0);

    showImage(image);
}

void testImageCompare()
{
    ImageComparator c;
    string refPath = RES "baboon1.jpg";
    Mat ref = imread(refPath);

    cout << "ref image is " << refPath <<endl;
    c.setRefrenceImage(ref);

    string s1 = RES "beach.jpg";
    Mat img1 = imread(s1);
    cout << s1 << " compare val is " << c.compare(img1) << endl;

    s1 = RES "boldt.jpg";
    img1 = imread(s1);
    cout << s1 << " compare val is " << c.compare(img1) << endl;

    s1 = RES "waves.jpg";
    img1 = imread(s1);
    cout << s1 << " compare val is " << c.compare(img1) << endl;

    s1 = RES "baboon3.jpg";
    img1 = imread(s1);
    cout << s1 << " compare val is " << c.compare(img1) << endl;

}

void testIntegral()
{
    // 打开图像
    Mat image = imread(RES "bike55.jpg");

    int xo=97, yo=112;
    int width=25, height=30;

    rectangle(image, Rect(xo, yo, width, height), Scalar(255, 0, 0), 1 , 1, 0);
    showImage(image);

    int64 start = getTickCount();
    // 定义图像的ROI（这里为骑自行车的女孩）
    Mat imageROI = image(Rect(xo, yo, width, height));
    Scalar sum = cv::sum(imageROI);
    cout << "sum use time :" << (getTickCount()-start)/getTickFrequency() << endl;

    start = getTickCount();
    // 计算积分图像
    Mat integralImage;
    cv::integral(image, integralImage, CV_32S);

    // 用三个加/减运算得到一个区域的累加值
    int sumInt= integralImage.at<int>(yo+height,xo+width)
            -integralImage.at<int>(yo+height,xo)
            -integralImage.at<int>(yo,xo+width)
            +integralImage.at<int>(yo,xo);
    cout << "sum(integral) use time :" << (getTickCount()-start)/getTickFrequency() << ", sum=" << sumInt << endl;

}

void testAdaptiveThresholding()
{
    Mat image = imread(RES "book.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    showImage(image);

    // 使用固定的阈值
    cv::Mat binaryFixed;
    cv::threshold(image,binaryFixed,70,255,cv::THRESH_BINARY);
    showImage(binaryFixed);

    Mat iimage;
    integral(image, iimage, CV_32S);

    //-------------------------
    int blockSize= 21; // 邻域的尺寸
    int threshold=10; // 像素将与(mean-threshold)进行比较
    int nl = iimage.rows;
    int nc = iimage.cols;
    // 逐行
    int halfSize= blockSize/2;
    for (int j=halfSize; j<nl-halfSize-1; j++) {
        // 得到第j行的地址
        uchar* data= image.ptr<uchar>(j);
        int* idata1= iimage.ptr<int>(j-halfSize);
        int* idata2= iimage.ptr<int>(j+halfSize+1);
        // 一个线条的像素
        for (int i=halfSize; i<nc-halfSize-1; i++) {
            // 计算累加值
            int sum= idata2[i+halfSize+1]
                    -idata2[i-halfSize]
                    -idata1[i+halfSize+1]
                    +idata1[i-halfSize];
            sum = sum/(blockSize*blockSize);
            // 应用自适应阈值
            if (data[i]<(sum-threshold))
                data[i]= 0;
            else
                data[i]=255;
        }
    }
    showImage(image);

    //------------
    Mat binaryAdaptive;
    cv::adaptiveThreshold(image,//输入图像
            binaryAdaptive,//输出二值图像
            255,//输出的最大值
            cv::ADAPTIVE_THRESH_MEAN_C,//方法(ADAPTIVE_THRESH_GAUSSIAN_C,高斯加权累计值要块一些)
            cv::THRESH_BINARY,//阀值类型
            blockSize,//块的大小
            threshold);//使用的阀值
    showImage(binaryAdaptive);

    //图像滤波方法
    Mat filtered;
    Mat binaryFiltered;
    boxFilter(image, filtered, CV_8U, Size(blockSize, blockSize));
    filtered = filtered - threshold;
    binaryFiltered = image>= binaryFiltered;
    showImage(binaryFiltered);
}

//转换成二值图层组成的多通道图像，一个平面关联直方图的一个箱子
//nPlanes必须是2的幂
void convertToBinaryPlanes(const Mat &input, Mat &output, int nPlanes)
{
    //需要屏蔽的位数
    int n = 8 - static_cast<uchar>(log(static_cast<double>(nPlanes))/log(2.0));
    //用来消除最低有效位的掩码
    uchar mask = 0xFF<<n;

    //创建二值图像的向量
    std::vector<Mat> planes;
    //消除最低有效位，箱子数量位nBins
    Mat reduced = input & mask;

    //计算每个二值平面图像
    for (int i = 0; i < nPlanes; ++i) {
        //将每个i<<shift的图像设为1
        planes.push_back((reduced==(i<<n)) & 0x1);
    }

    //创建多通道图像
    merge(planes, output);
}

void testIntegralTracking()
{
    // 打开图像
    Mat image = imread(RES "bike55.bmp", CV_LOAD_IMAGE_GRAYSCALE);

    int xo=97, yo=112;
    int width=25, height=30;
    Mat roi = image(Rect(xo, yo, width, height));

    //16个箱子的直方图
    Histogram1D h;
    h.setNBins(16);
    Mat refHistogram = h.getHistogram(roi);

    Mat secondImage = imread(RES "bike65.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    if (secondImage.empty()) {
        return;
    }
    //首先创建16个平面的二值图像
    Mat planes;
    convertToBinaryPlanes(secondImage, planes, 16);
    //然后计算积分图像
    IntegralImage<float, 16> intHistogram(planes);

    double maxSimilarity = 0.0;
    int xbest, ybest;
    Vec<float, 16> histogram;
    rectangle(secondImage, Rect(0, 110, secondImage.cols, height+10), Scalar(255, 255, 255));
    //遍历原始图像中女孩位置的水平长条
    for (int y = 110; y < 120; ++y) {
        for (int x = 0; x < secondImage.cols-width; ++x) {
            //用积分图像计算16个箱子的直方图
            histogram = intHistogram(x, y, width, height);
            //计算与基准直方图的差距
            double distance = compareHist(histogram, refHistogram, CV_COMP_INTERSECT);

            if (distance>maxSimilarity) {
                xbest = x;
                ybest = y;
                maxSimilarity = distance;
            }
        }
    }

    rectangle(secondImage, Rect(xbest, ybest, width, height), Scalar(0, 255, 0));
    showImage(secondImage);
}

void testErodeDilate()
{
    //    腐蚀图像相当于对其反色图像膨胀后再取反色；
    //    膨胀图像相当于对其反色图像腐蚀后再取反色。


    // 读取输入图像
    Mat image = imread(RES "binary.bmp");

    //openCV形态学函数支持就地处理

    // 腐蚀图像
    Mat eroded;
    erode(image, eroded, Mat());
    showImage(eroded, "Eroded", 1);

    // 膨胀图像
    Mat dilated;
    dilate(image, dilated, Mat());
    showImage(dilated, "Dilated", 1);


    //也可制定结构元素
    cv::Mat element(7,7,CV_8U,cv::Scalar(1));
    cv::erode(image,eroded,element);
    showImage(eroded, "Eroded2", 1);

    // 腐蚀图像三次
    cv::erode(image,eroded,cv::Mat(),cv::Point(-1,-1),3);
    //    cv::Point(-1,-1)表示原点是矩阵的中心点（默认值）
    showImage(eroded, "Eroded3");
}

void testMorphologyEx()
{
//    开启和闭合滤波器的定义，只是简单地使用了基本的腐蚀和膨胀运算。
//    闭合的定义是对图像先膨胀后腐蚀。
//    开启的定义是对图像先腐蚀后膨胀。
    Mat image = imread(RES "binary.bmp");

    cv::Mat element5(5,5,CV_8U,cv::Scalar(1));//结构大小依情况而定，5x5让效果更佳明显
    cv::Mat closed;
    cv::morphologyEx(image,closed,cv::MORPH_CLOSE,element5);
    showImage(closed, "MORPH_CLOSE", 1);

    cv::Mat opened;
    cv::morphologyEx(image,opened,cv::MORPH_OPEN,element5);
    showImage(opened, "MORPH_OPEN", 1);

    Mat result;
    // 膨胀原图像
    cv::dilate(image,result,cv::Mat());
    // 就地腐蚀膨胀后的图像
    cv::erode(result,result,cv::Mat());
    showImage(result, "MORPH_CLOSE 1");
}

void testMorphGradient()
{
    Mat image = imread(RES "building.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    // 用3 × 3结构元素得到梯度图像
    Mat result;
    cv::morphologyEx(image, result, cv::MORPH_GRADIENT, Mat());

    //对图像阀值化得到一个二值图像
    int threshold = 10;
    cv::threshold(result, result, threshold, 255, THRESH_BINARY);
    result ^= 0xFF;
    showImage(result);
}

void testMorphoCorners()
{
    Mat image = imread(RES "building.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    // 得到角点
    Mat corners;
    MorphoFeature morpho;
    morpho.setThreshold(40);
    corners = morpho.getCorners(image);
    showImage(corners, "Corners", 1);

    //在图像上显示角点
    morpho.drawOnImage(corners, image);
    showImage(image, "Corners On Image");

}

void testWatershedSegment()
{
    Mat image = imread(RES "group.jpg");
    if (image.empty()) {
        return;
    }
    showImage(image, 1);

    Mat binary = imread(RES "binaryGroup.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    showImage(binary, "Binary", 1);

    Mat fg;
    erode(binary, fg, Mat(), Point(-1,-1), 4);
    showImage(fg, "Forceground", 1);

    Mat bg;
    dilate(binary, bg, Mat(), Point(-1, -1), 4);
    threshold(bg, bg, 1, 128, THRESH_BINARY_INV);
    showImage(bg, "Background", 1);

    Mat markers(binary.size(), CV_8U, Scalar(0));
    markers = fg + bg;
    showImage(markers, "Marker", 1);


    // 创建分水岭分割类的对象
    WatershedSegmenter segmenter;
    // 设置标记图像，然后执行分割过程
    segmenter.setMarkers(markers);
    segmenter.process(image);

    Mat result = segmenter.getSegmentation();
    showImage(result, "Segmentation", 1);

    result = segmenter.getWatersheds();
    showImage(result, "Watersheds");
}

void testWatershedSegment1()
{
    Mat image = imread(RES "tower.jpg");

    Mat imageMask(image.size(), CV_8U, Scalar(0));

    rectangle(imageMask, Point(5, 5), Point(image.cols - 5, image.rows - 5), Scalar(255), 3);
    rectangle(imageMask, Point(image.cols/2 - 10, image.rows/2 - 10), Point(image.cols/2 + 10, image.rows/2 + 10), Scalar(1), 10);

//    rectangle(image, Point(5, 5), Point(image.cols - 5, image.rows - 5), Scalar(255, 255, 255), 3);
//    rectangle(image, Point(image.cols/2 - 10, image.rows/2 - 10), Point(image.cols/2 + 10, image.rows/2 + 10), Scalar(1, 1, 1), 10);
    showImage(image, WM_TAG, 1);

    WatershedSegmenter segmenter;
    segmenter.setMarkers(imageMask);
    segmenter.process(image);
    Mat result = segmenter.getWatersheds();
    showImage(result, "Watersheds");
}

void testMser()
{
    Mat image = imread(RES "building.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    if (image.empty()) {
        return;
    }
    dilate(image, image, Mat());
    showImage(image, 1);

    // 基本的MSER检测器
    MSER mser(5, // 检测极值区域时使用的增量
                  200, // 允许的最小面积
                  1500); // 允许的最大面积
    // 点集的容器
    vector<vector<Point>> points;
    // 检测MSER特征
    mser(image, points);//结果是一个包含若干个区域容器

    // 创建白色图像
    Mat output(image.size(), CV_8UC3);
    output = Scalar(255, 255, 255);
    // 随机数生成器
    cv::RNG rng;

    // 针对每个检测到的特征区域
    for (std::vector<std::vector<cv::Point>>::iterator it = points.begin();
         it != points.end(); it++ ) {
        // 生成随机颜色
        Vec3b c(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));


        for (std::vector<Point>::iterator itPts = it->begin();
             itPts != it->end(); itPts++ ) {
            //不重写MSER的像素
            if (output.at<cv::Vec3b>(*itPts)[0]==255) {
                output.at<cv::Vec3b>(*itPts)= c;
            }
        }
    }
    showImage(output, "MSER");
}

void testMserFeatures()
{
    Mat image = imread(RES "building.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    if (image.empty()) {
        return;
    }

    // 创建MSER特征检测器的实例
    MSERFeatures mserf(200, 1500, 0.5);
    std::vector<RotatedRect> rects;// 存放带边框的旋转矩形的容器
    Mat result = mserf.getImageOfEllipse(image, rects);// 检测并取得图像
    showImage(result);
}

void testGrabCut()
{
    Mat image = imread(RES "group.jpg");

    //定义一个带边框的矩形
    //矩形的外部被视为背景
    Rect rectangle(12, 100, 380, 170);
//    cv::rectangle(image, rectangle, Scalar(255, 255, 255));
    showImage(image);

    startCount();
    Mat result;//
    Mat fgModel, bgModel;//
    cv::grabCut(image,
                result,//分割结果
                rectangle,//包含前景的矩形
                bgModel, fgModel,//模型
                5,//迭代次数
                cv::GC_INIT_WITH_RECT);//使用矩形

    //GC_BGD，表示明确属于背景的像素
    //GC_FGD，表示明确属于前景的像素
    //GC_PR_BGD，表示可能属于背景的像素
    //GC_PR_FGD，表示可能属于前景的像素

    //取得“可能属于前景”的像素
    compare(result, GC_PR_FGD, result, CMP_EQ);
    Mat forceground(image.size(), CV_8UC3, Scalar(255, 255, 255));
    image.copyTo(forceground, result);//不复制背景图像
    cout << "testGrabCut use time : " << getTimeCount() << endl;

    //用按位与运算检查第一位，因为（GC_FGD，GC_PR_FGD）定义的值位1, 3
    result &= 1;//如果是前景色，结果位1
    showImage(forceground, "Forceground");
}

int main(int argc, char *argv[])
{

#ifdef SHOW_WIN_FORM

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();

#endif


    //    osxHello();
    
    //    test1();
    //    test2();
    //    testROI();
    //    testRoiMask();
    //    testSalt();
    //    testColorReduce(5);
    //    testSharpen(1);

    //    testImgCalc(4);
    //    testSplit();
    //    testRemap();
    
    //    testColorDetector(1);
    //    testHSV();
    //    testDetectHSV();

    //    testHistogram1D();
    //    testLut();
    //    testStrech();

    //    testBackProject();
    //    testBackProject1();
    //    testHsvBackProject();

    //    testImageCompare();

    //    testIntegral();
    //    testAdaptiveThresholding();
    //    testIntegralTracking();

    //Chapter:05
    //    testErodeDilate();
    //    testMorphologyEx();

    //    testMorphGradient();
    //    testMorphoCorners();

    //    testWatershedSegment();
    //    testWatershedSegment1();

    //    testMser();
    //    testMserFeatures();

    //    testGrabCut();

    //Chapter:06

    return 0;
}
