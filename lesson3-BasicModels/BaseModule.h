#pragma once
#ifndef BASEMODULE_H
#define BASEMODULE_H

#endif // BASEMODULE_H
#undef slots
#include <torch/torch.h>
#include<torch/script.h>
#define slots Q_SLOTS

inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 0, bool with_bias = false) {
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    conv_options.padding(padding);
    conv_options.bias(with_bias);
    return conv_options;
}

class ConvReluBnImpl : public torch::nn::Module {
public:
    ConvReluBnImpl(int input_channel=3, int output_channel=64, int kernel_size = 3, int stride = 1);
    torch::Tensor forward(torch::Tensor x);
private:
    // Declare layers
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };
};
TORCH_MODULE(ConvReluBn);

class LinearBnReluImpl : public torch::nn::Module{
public:
    LinearBnReluImpl(int intput_features, int output_features);
    torch::Tensor forward(torch::Tensor x);
private:
    //layers
    torch::nn::Linear ln{nullptr};
    torch::nn::BatchNorm1d bn{nullptr};
};
TORCH_MODULE(LinearBnRelu);
