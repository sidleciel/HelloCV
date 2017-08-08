#ifndef COLORDETECTCONTROLLER_H
#define COLORDETECTCONTROLLER_H

#include <colordetector.h>

class ColorDetectController
{
private:
    ColorDetector *cdetector;
    Mat image;//要处理的图像
    Mat result;//处理结果
public:
    ColorDetectController();
    ~ColorDetectController();

    void setColorDistanceThreshold(int distance);
    int getColorDistanceThreshold() const;
    void setTargetColor(unsigned char blue, unsigned char green, unsigned char red);
    void getTargetColor(unsigned char &blue, unsigned char &green, unsigned char &red);

    bool setInputImage(string filename);
    Mat getInputImage() const;
    void process();
    const Mat getLastResult() const;
};

#endif // COLORDETECTCONTROLLER_H
