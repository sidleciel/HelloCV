#include "colordetectcontroller.h"

ColorDetectController::ColorDetectController()
{
    cdetector = new ColorDetector();
}

ColorDetectController::~ColorDetectController()
{
    delete cdetector;
}

void ColorDetectController::setColorDistanceThreshold(int distance)
{
    cdetector->setColorDistanceThreshold(distance);
}

int ColorDetectController::getColorDistanceThreshold() const
{
    return cdetector->getColorDistanceThreshold();
}

void ColorDetectController::setTargetColor(unsigned char blue, unsigned char green, unsigned char red)
{
    cdetector->setTargetColor(blue,green,red);
}

void ColorDetectController::getTargetColor(unsigned char &blue, unsigned char &green, unsigned char &red)
{
    Vec3b color = cdetector->getTargetColor();
    blue = color[0];
    green = color[1];
    red = color[2];
}

bool ColorDetectController::setInputImage(string filename)
{
    image = imread(filename);
    return !image.empty();
}

Mat ColorDetectController::getInputImage() const
{
    return image;
}

void ColorDetectController::process()
{
    result = cdetector->cvProcess(image);
}

const Mat ColorDetectController::getLastResult() const
{
    return result;
}
