#ifndef UNET_H
#define UNET_H
#include"ResNet.h"
#include"UNetDecoder.h"

class UNetImpl : public torch::nn::Module
{
public:
    UNetImpl(int num_classes, std::string encoder_name = "resnet18", std::string pretrained_path = "", int encoder_depth = 5,
             std::vector<int> decoder_channels={256, 128, 64, 32, 16}, bool use_attention = false);
    torch::Tensor forward(torch::Tensor x);
private:
    ResNet encoder{nullptr};
    UNetDecoder decoder{nullptr};
    SegmentationHead segmentation_head{nullptr};
    int num_classes = 1;
    std::vector<int> BasicChannels = {3, 64, 64, 128, 256, 512};
    std::vector<int> BottleChannels = {3, 64, 256, 512, 1024, 2048};
    std::map<std::string, std::vector<int>> name2layers = getParams();
};TORCH_MODULE(UNet);

#endif // UNET_H
