#include "UNet.h"

UNetImpl::UNetImpl(int _num_classes, std::string encoder_name, std::string pretrained_path, int encoder_depth,
                   std::vector<int> decoder_channels, bool use_attention){
    num_classes = _num_classes;
    std::vector<int> encoder_channels = BasicChannels;
    if(!name2layers.count(encoder_name)) throw "encoder name must in {resnet18, resnet34, resnet50, resnet101}";
    if(encoder_name!="resnet18" && encoder_name!="resnet34"){
        encoder_channels = BottleChannels;
    }

    encoder = pretrained_resnet(1000, encoder_name, pretrained_path);
    decoder = UNetDecoder(encoder_channels,decoder_channels, encoder_depth, use_attention, false);
    segmentation_head = SegmentationHead(decoder_channels[decoder_channels.size()-1], num_classes, 1, 1);

    register_module("encoder",encoder);
    register_module("decoder",decoder);
    register_module("segmentation_head",segmentation_head);
}

torch::Tensor UNetImpl::forward(torch::Tensor x){
    std::vector<torch::Tensor> features = encoder->features(x);
    x = decoder->forward(features);
    x = segmentation_head->forward(x);
    return x;
}
