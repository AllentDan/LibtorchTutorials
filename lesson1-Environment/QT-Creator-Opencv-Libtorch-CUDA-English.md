---
title: QT Creator + Opencv + Libtorch +CUDA English
date: 2020-03-05 20:07:15
tags:
---
1. Download and install [QT Creator](http://download.qt.io/archive/qtcreator/), configure the environment. Note that the MSVC compiler component is checked during installation. Use MSVC to compile the project. Configure MSVC 2017 x64 in Tools->Options->Build Kits->MSVC 2017 x64, and select the c and c++ compilers as amd64.

2. If the computer does not have the cdb.exe file (Search by Everything), [download and install it](https://developer.microsoft.com/en-US/windows/downloads/windows-10-sdk). After installation, select Tools->Options->Build Kits->MSVC 2017 x64->Debugger (Debugger) and add cdb.exe.

3. Download [OpenCV](https://opencv.org/releases/) and [libtorch](https://pytorch.org/get-started/locally/). Configure the correct path to the project's .pro file and add it at the end of the .pro file
```
INCLUDEPATH += your path to\opencv-4.5.0-vc14_vc15\opencv\build\include \
your path to\libtorch17release\include \
your path to\libtorch17release\include\torch\csrc\api\include

LIBS += -Lyour path to\opencv-4.5.0-vc14_vc15\opencv\build\x64\vc15\lib -lopencv_world450 \
-Lyour path to\libtorch17release\lib -lc10 -ltorch -lc10_cuda -lcaffe2_detectron_ops_gpu -lc10d -ltorch_cpu \
-ltorch_cuda -lgloo -lcaffe2_module_test_dynamic -lasmjit -lcaffe2_nvrtc -lclog -lcpuinfo -ldnnl -lfbgemm -lgloo_cuda \
-lmkldnn -INCLUDE:?warp_size@cuda@at@@YAHXZ
```

4. The project is configured in Release mode, qmake is successfully run, right-click the project to rebuild. Possible errors are:
   Syntax error: identifier “IValue”...
   change the codes in #include \<torch/torch.h\> to:
```cpp
#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS
```
if /libtorch/include/ATen/core/ivalue.h and IValue_init.h throw errors, comment the following three lines
```cpp
/// \cond DOXYGEN_CANNOT_HANDLE_CONSTRUCTORS_WITH_MACROS_SO_EXCLUDE_THIS_LINE_FROM_DOXYGEN
C10_DEPRECATED_MESSAGE("IValues based on std::vector<T> are potentially slow and deprecated. Please use c10::List<T> instead.")
/// \endcond
```

1. The main.cpp for testing：
```cpp
#include "mainwindow.h"
#include<opencv2/opencv.hpp>
#include <QApplication>
#include<iostream>
#undef slots
#include<torch/script.h>
#include<torch/torch.h>
#define slots Q_SLOTS

class ConvReluBnImpl : public torch::nn::Module {
public:
    ConvReluBnImpl(int input_channel=3, int output_channel=64, int kernel_size = 3);
    torch::Tensor forward(torch::Tensor x);
private:
    // Declare layers
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };
};
TORCH_MODULE(ConvReluBn);

ConvReluBnImpl::ConvReluBnImpl(int input_channel, int output_channel, int kernel_size) {
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channel, output_channel, kernel_size).padding(1)));
    bn = register_module("bn", torch::nn::BatchNorm2d(output_channel));

}

torch::Tensor ConvReluBnImpl::forward(torch::Tensor x) {
    x = torch::relu(conv->forward(x));
    x = bn(x);
    return x;
}

int main(int argc, char *argv[])
{
    //test torch
    auto device = torch::Device(torch::kCUDA);
    auto model = ConvReluBn(3,4,3);
    model->to(device);
    auto input = torch::zeros({1,3,12,12},torch::kFloat).to(device);
    auto output = model->forward(input);
    std::cout<<output.sizes()<<std::endl;

    //test opencv
    cv::Mat image = cv::imread("C:\\Users\\Administrator\\Pictures\\1.jpg");
    cv::Mat M(200, 200, CV_8UC3, cv::Scalar(0, 0, 255));
    if(!M.data)
        return 0;
    cv::imshow("fff",image);
    cv::imshow("ddd",M);
    cv::waitKey(0);
    cv::destroyAllWindows();
    //test qt
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}

```