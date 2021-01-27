#ifndef CNN_H
#define CNN_H

#endif // CNN_H
#include"BaseModule.h"

class plainCNN : public torch::nn::Module{
public:
    plainCNN(int in_channels, int out_channels);
    torch::Tensor forward(torch::Tensor x);
private:
    int mid_channels[3] = {32,64,128};
    ConvReluBn conv1{nullptr};
    ConvReluBn down1{nullptr};
    ConvReluBn conv2{nullptr};
    ConvReluBn down2{nullptr};
    ConvReluBn conv3{nullptr};
    ConvReluBn down3{nullptr};
    torch::nn::Conv2d out_conv{nullptr};
};
