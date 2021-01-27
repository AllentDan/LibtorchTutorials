#include<CNN.h>

plainCNN::plainCNN(int in_channels, int out_channels){
    conv1 = ConvReluBn(in_channels,mid_channels[0],3);
    down1 = ConvReluBn(mid_channels[0],mid_channels[0],3,2);
    conv2 = ConvReluBn(mid_channels[0],mid_channels[1],3);
    down2 = ConvReluBn(mid_channels[1],mid_channels[1],3,2);
    conv3 = ConvReluBn(mid_channels[1],mid_channels[2],3);
    down3 = ConvReluBn(mid_channels[2],mid_channels[2],3,2);
    out_conv = torch::nn::Conv2d(conv_options(mid_channels[2],out_channels,3));

    conv1 = register_module("conv1",conv1);
    down1 = register_module("down1",down1);
    conv2 = register_module("conv2",conv2);
    down2 = register_module("down2",down2);
    conv3 = register_module("conv3",conv3);
    down3 = register_module("down3",down3);
    out_conv = register_module("out_conv",out_conv);
}

torch::Tensor plainCNN::forward(torch::Tensor x){
    x = conv1->forward(x);
    x = down1->forward(x);
    x = conv2->forward(x);
    x = down2->forward(x);
    x = conv3->forward(x);
    x = down3->forward(x);
    x = out_conv->forward(x);
    return x;
}
