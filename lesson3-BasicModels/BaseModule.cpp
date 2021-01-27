#include"BaseModule.h"

ConvReluBnImpl::ConvReluBnImpl(int input_channel, int output_channel, int kernel_size, int stride) {
    conv = register_module("conv", torch::nn::Conv2d(conv_options(input_channel,output_channel,kernel_size,stride,kernel_size/2)));
    bn = register_module("bn", torch::nn::BatchNorm2d(output_channel));

}

torch::Tensor ConvReluBnImpl::forward(torch::Tensor x) {
    x = torch::relu(conv->forward(x));
    x = bn(x);
    return x;
}

LinearBnReluImpl::LinearBnReluImpl(int in_features, int out_features){
    ln = register_module("ln", torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features)));
    bn = register_module("bn", torch::nn::BatchNorm1d(out_features));
}

torch::Tensor LinearBnReluImpl::forward(torch::Tensor x){
    x = torch::relu(ln->forward(x));
    x = bn(x);
    return x;
}
