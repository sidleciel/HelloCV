#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <colordetectcontroller.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

//    void openImage();
//    void process();
//    void cancel();
//    void ok();

    void displayImage(const Mat &image);

private slots:
    void on_btnOpenImage_clicked();

    void on_btnProcess_clicked();

    void on_btnOK_clicked();

    void on_btnCancel_clicked();

private:
    Ui::MainWindow *ui;

    ColorDetectController *cdetectcontroller;
};

#endif // MAINWINDOW_H
