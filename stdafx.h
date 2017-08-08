#ifndef STDAFX_H
#define STDAFX_H

#include <QtGlobal>

#ifdef Q_OS_MAC
#define RES "/Users/xietao/Downloads/qtProjects/helloCv/img/"
#endif

#ifdef Q_OS_WIN
#include <windows.h>
#define RES "E:\\workspace.qt\\HelloCv\\img\\"
#endif

#define WM_TAG "Original Image"

//#define SHOW_WIN_FORM

#endif // STDAFX_H
