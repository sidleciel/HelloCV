#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "QFileDialog"
#include "QTextCodec"
#include "QMessageBox"

#include <iostream>
using namespace std;

#include <stdafx.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    cdetectcontroller = new ColorDetectController();

    //    connect(ui->btnOpenImage,SIGNAL(clicked(bool)),this,SLOT(openImage()));
    //    connect(ui->btnProcess,SIGNAL(clicked(bool)),this,SLOT(process()));
    //    connect(ui->btnOK,SIGNAL(clicked(bool)),this,SLOT(ok()));
    //    connect(ui->btnCancel,SIGNAL(clicked(bool)),this,SLOT(cancel()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::displayImage(const Mat &image)
{
    if (image.empty()) {
        cout<<"Image is empty."<<endl;
        return;
    }

    Mat rgb;
    QImage img;

    cout<<"image channels="<<image.channels()<<endl;
    if (image.channels()==3) {
        cvtColor(image,rgb,CV_BGR2RGB);

        img = QImage((const unsigned char*)rgb.data,
                     rgb.cols, rgb.rows, rgb.cols*rgb.channels(),
                     QImage::Format_RGB888);
    } else {
        img = QImage((const unsigned char*)image.data,
                     image.cols,image.rows,image.step,
                     QImage::Format_Grayscale8);
    }

    ui->labelImage->setPixmap(QPixmap::fromImage(img).scaled(ui->labelImage->size()));//setPixelmap(QPixmap::fromImage(img));
    //    ui->labelImage->setPixmap(QPixmap::fromImage(img));
    ui->labelImage->resize(ui->labelImage->pixmap()->size());//resize(ui->label->pixmap()->size());
}

void MainWindow::on_btnOpenImage_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"),RES,tr("Image File(*.jpg *.bmp *.png)"));
    QTextCodec *code = QTextCodec::codecForName("GB2312");

    string name = code->fromUnicode(filename).data();

    cout<<"Image src="<<name<<endl;
    if(cdetectcontroller->setInputImage(name)){
        cout<<"Image opened."<<endl;

        displayImage(cdetectcontroller->getInputImage());
    }else{
        cout<<"Image open failed."<<endl;
    }
}

void MainWindow::on_btnProcess_clicked()
{
    cdetectcontroller->process();
}

void MainWindow::on_btnOK_clicked()
{
    Mat image(cdetectcontroller->getLastResult());
    displayImage(image);
}

void MainWindow::on_btnCancel_clicked()
{
    this->close();
}
