#ifndef VIDEOPROCESSOR_H
#define VIDEOPROCESSOR_H

#include "iostream"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

class VideoProcessor
{
private:
    // OpenCV视频捕获对象
    cv::VideoCapture capture;
    // 处理每一帧时都会调用的回调函数
    void (*process)(cv::Mat&, cv::Mat&);
    // 布尔型变量，表示该回调函数是否会被调用
    bool callIt;
    // 输入窗口的显示名称
    std::string windowNameInput;
    // 输出窗口的显示名称
    std::string windowNameOutput;
    // 帧之间的延时
    int delay;
    // 已经处理的帧数
    long fnumber;
    // 读取内容可能为视频流
    bool isFile;
    // 达到这个帧数时结束
    long frameToStop;
    // 结束处理
    bool stop;

public:
    VideoProcessor();

    void setFrameProcessor(void (*frameProcessingCallback)(cv::Mat&, cv::Mat&));
    bool setInput(std::string filename, bool isFile = true);
    bool setInput(int id);
    void displayInput(std::string wn);
    void displayOutput(std::string wn);
    void run();
    void stopIt();
    bool isStopped();
    bool isOpened();
    void setDelay(int d);
    bool readNextFrame(cv::Mat& frame);
    void callProcess();
    void dontCallProcess();
    void stopAtFrameNo(long frame);
    long getFrameNumber();
    long getFrameRate();
};

#endif // VIDEOPROCESSOR_H
