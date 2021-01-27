#include "mainwindow.h"
#include<opencv2/opencv.hpp>
#include <QApplication>
#include<iostream>
#undef slots
#include<torch/script.h>
#include<torch/torch.h>
#define slots Q_SLOTS
#include<vector>

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
    auto b = torch::rand({3,4});
    auto c = torch::rand({3,4});
    std::cout<<b<<c<<b*c<<b/c<<b.mm(c.t());
    auto cat = torch::cat({b,c},1);
    auto stack = torch::stack({b,c},1);
    std::cout<<b<<c<<cat<<stack;

    //test torch
    auto device = torch::Device(torch::kCUDA);
    auto model = ConvReluBn(3,4,3);
    model->to(device);
    auto input = torch::zeros({1,3,12,12},torch::kFloat).to(device);
    auto output = model->forward(input);
    std::cout<<output<<std::endl;

    //test opencv
    cv::Mat image = cv::imread("C:\\Users\\Administrator\\Pictures\\1.jpg");
    cv::Mat M(200, 200, CV_8UC3, cv::Scalar(0, 0, 255));
    if(!M.data)
        return 0;
    cv::imshow("fff",image);
    cv::imshow("ddd",M);
    cv::waitKey(0);
    cv::destroyAllWindows();
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
