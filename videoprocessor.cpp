#include "videoprocessor.h"

VideoProcessor::VideoProcessor()
{

}

// 设置针对每一帧调用的回调函数
void VideoProcessor::setFrameProcessor(void (*frameProcessingCallback)(cv::Mat&, cv::Mat&)) {
    process = frameProcessingCallback;
}

// 设置视频文件的名称
bool VideoProcessor::setInput(std::string filename, bool isFile) {
    fnumber= 0;
    this->isFile = isFile;
    // 防止已经有资源与VideoCapture实例关联
    if(capture.isOpened())
        capture.release();
    // 打开视频文件
    return capture.open(filename);
}

// 设置视频文件的名称
bool VideoProcessor::setInput(int id) {
    fnumber= 0;
    this->isFile = false;
    // 防止已经有资源与VideoCapture实例关联
    capture.release();
    // 打开视频文件
    return capture.open(id);
}

// 用于显示输入的帧
void VideoProcessor::displayInput(std::string wn) {
    windowNameInput= wn;
    cv::namedWindow(windowNameInput);
}
// 用于显示处理过的帧
void VideoProcessor::displayOutput(std::string wn) {
    windowNameOutput= wn;
    cv::namedWindow(windowNameOutput);
}

// 抓取（并处理）序列中的帧
void VideoProcessor::run() {
    // 当前帧
    cv::Mat frame;
    // 输出帧
    cv::Mat output;
    // 如果没有设置捕获设备
    if (!isOpened())
        return;
    stop= false;
    while (!isStopped()) {
        // 读下一帧（如果有）
        if (!readNextFrame(frame))
            break;
        // 显示输入的帧
        if (windowNameInput.length()!=0)
            cv::imshow(windowNameInput,frame);
        // 调用处理函数
        if (callIt && !process) {
            // 处理帧
            process(frame, output);
            // 递增帧数
            fnumber++;
        } else { // 没有处理
            output= frame;
        }
        // 显示输出的帧
        if (windowNameOutput.length()!=0)
            cv::imshow(windowNameOutput,output);
        // 产生延时
        if (delay>=0 && cv::waitKey(delay)>=0){
            stopIt();
        }
        // 检查是否需要结束
        if (isFile && frameToStop>=0 && getFrameNumber()==frameToStop)
            stopIt();
    }
}


// 结束处理
void VideoProcessor::stopIt() {
    std::cout << "stopIt" << std::endl;
    stop= true;
}
// 处理过程是否已经停止？
bool VideoProcessor::isStopped() {
    return stop;
}
// 捕获设备是否已经打开？
bool VideoProcessor::isOpened() {
    capture.isOpened();
}
// 设置帧之间的延时，
// 0表示每一帧都等待，
// 负数表示不延时
void VideoProcessor::setDelay(int d) {
    delay= d;
}

// 取得下一帧，
// 可以是：视频文件或者摄像机
bool VideoProcessor::readNextFrame(cv::Mat& frame) {
    return capture.read(frame);
}

// 需要调用回调函数process
void VideoProcessor::callProcess() {
    callIt= true;
}
// 需要调用回调函数process
void VideoProcessor::dontCallProcess() {
    callIt= false;
}

void VideoProcessor::stopAtFrameNo(long frame) {
    frameToStop= frame;
}

long VideoProcessor::getFrameRate()
{
    return static_cast<long>(capture.get(CV_CAP_PROP_FPS));
}

// 返回下一帧的编号
long VideoProcessor::getFrameNumber() {

    //    CV_CAP_PROP_FPS 标志获得帧速率
    //    CV_CAP_PROP_FRAME_COUNT 获得视频文件的总帧数（整数）
    //    CV_CAP_PROP_POS_FRAMES 让视频跳转到指定的帧
    //    CV_CAP_PROP_POS_MSEC 以毫秒为单位指定位置
    //    CV_CAP_PROP_POS_AVI_RATIO 指定视频内部的相对位置（0.0表示视频开始位置，1.0表示结束位置）

    // 跳转到第100帧
    //    double position= 100.0;
    //    capture.set(CV_CAP_PROP_POS_FRAMES, position);

    // 从捕获设备获取信息
    long fnumber = static_cast<long>(capture.get(CV_CAP_PROP_POS_FRAMES));
    return fnumber;
}
