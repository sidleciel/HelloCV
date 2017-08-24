#-------------------------------------------------
#
# Project created by QtCreator 2017-08-06T23:02:22
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = helloCv
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp \
    colordetector.cpp \
    colordetectcontroller.cpp \
    histogram1d.cpp \
    colorhistogram.cpp \
    contentfinder.cpp \
    imagecomparator.cpp \
    integralimage.cpp \
    morphofeature.cpp

HEADERS += \
        mainwindow.h \
    colordetector.h \
    colordetectcontroller.h \
    histogram1d.h \
    stdafx.h \
    colorhistogram.h \
    contentfinder.h \
    imagecomparator.h \
    integralimage.h \
    morphofeature.h

FORMS += \
        mainwindow.ui

win32 {

INCLUDEPATH += \
        D:\opencv\opencv_qt\install\include \
        D:\opencv\opencv_qt\install\include\opencv \
        D:\opencv\opencv_qt\install\include\opencv2

LIBS += \
    D:\opencv\opencv_qt\install\x86\mingw\lib\libopencv_core2413.dll.a \
    D:\opencv\opencv_qt\install\x86\mingw\lib\libopencv_highgui2413.dll.a \
    D:\opencv\opencv_qt\install\x86\mingw\lib\libopencv_imgproc2413.dll.a \
    D:\opencv\opencv_qt\install\x86\mingw\lib\libopencv_video2413.dll.a

}

macx {

INCLUDEPATH += \
        /usr/local/include \
        /usr/local/include/opencv \
        /usr/local/include/opencv2

LIBS += \
    -L/usr/local/lib \
    -lopencv_core \
    -lopencv_highgui \
    -lopencv_imgproc \
    -lopencv_video

}
