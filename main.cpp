//#include "mainwindow.h"
//#include <QApplication>

//int main(int argc, char *argv[])
//{
//    QApplication a(argc, argv);
//    MainWindow w;
//    w.show();

//    return a.exec();
//}


#include <QtGlobal>

#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#include <windows.h>

#ifdef Q_OS_WIN
#define RES "E:\\workspace.qt\\HelloCv\\img\\"
#elif Q_OS_MAC
#define RES "/Users/xietao/Downloads/qtProjects/helloCv/"
#else
#define RES ""
#endif


#define WM_TAG "Original Image"

void flipSave(Mat &img){
    //flip����ת
    Mat result;
    int flipCode = -1;//����ˮƽ��0ˮƽ�ʹ�ֱ����ת��������ֱ��ת
    flip(img, result, flipCode);
    imshow(WM_TAG, result);

    //����
    string output = "output.png";//���ݺ�׺�����ʽ
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
           Point(img.cols / 2, img.rows / 2),//Բ��
           100,//�뾶
           Scalar(191),//��ɫ
           10);//���
    imshow(WM_TAG, img);

    putText(img,
            "This is HaiMa.",
            Point(img.cols / 2 - 110, img.rows / 2 + 150),
            FONT_HERSHEY_PLAIN,//����
            2.0,//�����С
            255,
            2//���
            );
    imshow(WM_TAG, img);
    waitKey(0);
}


//���Ժ�����������һ��ͼ��
Mat funciton(){
    //����ͼ��
    Mat ima(500, 500, CV_8UC(1), 50);
    return ima;//����ͼ��
}

void test2(){
    string WM_T = "Image";
    string WM_T1 = "Image 1";
    string WM_T2 = "Image 2";
    string WM_T3 = "Image 3";
    string WM_T4 = "Image 4";
    string WM_T5 = "Image 5";
    //���崰��
    namedWindow(WM_T1);
    namedWindow(WM_T2);
    namedWindow(WM_T3);
    namedWindow(WM_T4);
    namedWindow(WM_T5);
    namedWindow(WM_T);

    //����һ��240��X320�е�ͼ��
    Mat image1(240, 320, CV_8U, 100);
    imshow(WM_T, image1);//��ʾͼ��
    waitKey(0);//�ȴ�����

    //���·���һ���µ�ͼ��
    image1.create(200, 200, CV_8U);
    image1 = 200;

    imshow(WM_T, image1);
    waitKey(0);

    //����һ����ɫͼ��
    //ͨ����������ΪBGR
    Mat image2(240, 320, CV_8UC3, Scalar(0, 0, 255));
    Mat image5(240, 320, CV_8UC3, Scalar(0, 0, 255));
    //���ߣ�
    //Mat image2(240, 320, CV_8UC3);
    //image2 = Scalar(0, 0, 255);

    imshow(WM_T2, image2);
    waitKey(0);

    //����һ��ͼ��
    Mat image3 = imread("puppy.jpg");

    //������Щͼ��ָ��ͬһ�����ݿ�
    Mat image4(image3);
    image1 = image3;

    //��Щ��Դͼ��ĸ���ͼ��
    image3.copyTo(image2);
    image5 = image3.clone();

    //image1.release();
    //image4.release();

    //ת��ͼ����������
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

    //����2
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

    // ��ͼ������½Ƕ���һ��ROI
    Mat imageROI = image(cv::Rect(image.cols - logo.cols,
                                  image.rows - logo.rows,
                                  logo.cols, logo.rows));
    // �ѱ�־��Ϊ���루�����ǻҶ�ͼ��
    cv::Mat mask(logo);
    // �����־��ֻ�������벻Ϊ0��λ��
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
    int nl = image.rows;//����
    int nc = image.cols*image.channels();//ÿ�е�������

    for (int j = 0; j < nl; j++)
    {
        //ȡ��i�еĵ�ַ
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++)
        {
            //����ÿ������
            //data[i] = data[i] / div*div + div / 2;//��������

            data[i] = data[i] - data[i] % div + div / 2;//ȡģ�㷨
            //���ش������
        }//һ�н���
    }
}

void colorReduce1(Mat image, int n = 4){
    int nl = image.rows;//����
    int nc = image.cols*image.channels();//ÿ�е�������

    int div = pow(2, n);
    uchar mask = 0xFF << n;

    for (int j = 0; j < nl; j++)
    {
        //ȡ��i�еĵ�ַ
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++)
        {
            //����ÿ������
            *data &= mask;// ����
            *data++ += div >> 1;// ����div/2
            //���ش������
        }//һ�н���
    }
}

void colorReduce2(Mat &image, int div = 64)
{//������ͼ��ĸ�Чɨ��
    int nl = image.rows;//����
    int nc = image.cols*image.channels();

    if(image.isContinuous())
    {
        cout<<"This image is continous."<<endl;
        nc *= nl;
        nl = 1;//����һ����һά����

        //û���������
        image.reshape(1,1);//�µ�ͨ�������µ�����
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
{//�õ�����ɨ��ͼ��
    cout << "�õ�����ɨ��ͼ��" << endl;
    //    ����������
    //    cv::MatConstIterator_<cv::Vec3b> it;
    //    ����
    //    cv::Mat_<cv::Vec3b>::const_iterator it;

    //��ȡ��ʼλ�õ�����
    Mat_<Vec3b>::iterator it = image.begin<Vec3b>();
    //��ȡ����λ�õ�����
    Mat_<Vec3b>::iterator itend = image.end<Vec3b>();

    //    ����2������Ҫ�ƶ����ص���������
    Mat_<Vec3b> cimage(image);
    it = cimage.begin();
    itend = cimage.end();

    //ѭ��������������
    for(;it!=itend;++it)
    {
        (*it)[0] = (*it)[0]/div*div + div/2;
        (*it)[1] = (*it)[1]/div*div + div/2;
        (*it)[2] = (*it)[2]/div*div + div/2;
    }
}


void testColorReduce(){
    Mat img = imread(RES "boldt.jpg");

    if(img.empty())
    {
        cout<<(RES "boldt.jpg") << endl <<"This image is empty."<<endl;
        return;
    }

    const int64 start = getTickCount();
    colorReduce3(img);
    double duration = (getTickCount()-start)/getTickFrequency();
    cout<<"colorReduce duration="<<duration<<endl;


    namedWindow("Image");
    imshow("Image", img);

    waitKey(0);
}

void* ptrTest(Mat &image)
{//�Ͳ��ָ���㷨���˷������Ƽ�ʹ��
    uchar* data = image.data;//��ȡͼ����׵�ַ

    //image.step ��ȡһ�����ص����ֽ���������������أ�
    data += image.step;//��һ�еĵ�ַ

    //(j,i)�����صĵ�ַ����image.at(j,i)
    int j,i;
    data = image.data + image.step * j + image.elemSize()*i;
}

void sharpen(const Mat &image, Mat &result)
{
    result.create(image.size(),image.type());
    int nchannels = image.channels();

    //    �����񻯵���ֵ��
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


void testSharpen(){
    Mat img = imread(RES "boldt.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    if(img.empty())
    {
        cout<<(RES "boldt.jpg") << endl <<"This image is empty."<<endl;
        return;
    }

    const int64 start = getTickCount();
    Mat result;
    sharpen(img, result);
    double duration = (getTickCount()-start)/getTickFrequency();
    cout<<"sharpen duration="<<duration<<endl;


    namedWindow("Image");
    imshow("Image", result);

    waitKey(0);
}

int osx_main(int argc, char *argv[])
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
    return 0;
}


int main(int argc, char *argv[])
{
#ifdef Q_OS_MAC
    osx_main(argc, argv[]);

#elif defined(Q_OS_WIN)

    //    test1();
    //    test2();
    //    testROI();
    //    testRoiMask();
    //    testSalt();
//    testColorReduce();
    testSharpen();

#endif

    return 0;
}
