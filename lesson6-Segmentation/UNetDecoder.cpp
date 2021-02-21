#include "UNetDecoder.h"

SCSEModuleImpl::SCSEModuleImpl(int in_channels, int reduction, bool _use_attention){
    use_attention = _use_attention;
    cSE = torch::nn::Sequential(
            torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)),
            torch::nn::Conv2d(conv_options(in_channels, in_channels / reduction, 1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Conv2d(conv_options(in_channels / reduction, in_channels, 1)),
            torch::nn::Sigmoid());
    sSE = torch::nn::Sequential(torch::nn::Conv2d(conv_options(in_channels, 1, 1)), torch::nn::Sigmoid());
    register_module("cSE",cSE);
    register_module("sSE",sSE);
}

torch::Tensor SCSEModuleImpl::forward(torch::Tensor x){
    if(!use_attention) return x;
    return x * cSE->forward(x) + x * sSE->forward(x);
}

Conv2dReLUImpl::Conv2dReLUImpl(int in_channels, int out_channels, int kernel_size, int padding){
    conv2d = torch::nn::Conv2d(conv_options(in_channels,out_channels,kernel_size,1,padding));
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));
    register_module("conv2d", conv2d);
    register_module("bn", bn);
}

torch::Tensor Conv2dReLUImpl::forward(torch::Tensor x){
    x = conv2d->forward(x);
    x = bn->forward(x);
    return x;
}

DecoderBlockImpl::DecoderBlockImpl(int in_channels, int skip_channels, int out_channels, bool skip, bool attention){
    conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, 3, 1);
    conv2 = Conv2dReLU(out_channels, out_channels, 3, 1);
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    upsample = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kNearest));

    attention1 = SCSEModule(in_channels + skip_channels, 16, attention);
    attention2 = SCSEModule(out_channels, 16, attention);
    register_module("attention1", attention1);
    register_module("attention2", attention2);
    is_skip = skip;
}

torch::Tensor DecoderBlockImpl::forward(torch::Tensor x, torch::Tensor skip){
    x = upsample->forward(x);
    if (is_skip){
        x = torch::cat({x, skip}, 1);
        x = attention1->forward(x);
    }
    x = conv1->forward(x);
    x = conv2->forward(x);
    x = attention2->forward(x);
    return x;
}

torch::nn::Sequential CenterBlock(int in_channels, int out_channels){
    return torch::nn::Sequential(Conv2dReLU(in_channels, out_channels, 3, 1),
                                 Conv2dReLU(out_channels, out_channels, 3, 1));
}

UNetDecoderImpl::UNetDecoderImpl(std::vector<int> encoder_channels, std::vector<int> decoder_channels, int n_blocks,
                         bool use_attention, bool use_center)
{
    if (n_blocks != decoder_channels.size()) throw "Model depth not equal to your provided `decoder_channels`";
    std::reverse(std::begin(encoder_channels),std::end(encoder_channels));

    // computing blocks input and output channels
    int head_channels = encoder_channels[0];
    std::vector<int> out_channels = decoder_channels;
    decoder_channels.pop_back();
    decoder_channels.insert(decoder_channels.begin(),head_channels);
    std::vector<int> in_channels = decoder_channels;
    encoder_channels.erase(encoder_channels.begin());
    std::vector<int> skip_channels = encoder_channels;
    skip_channels[skip_channels.size()-1] = 0;

    if(use_center)  center = CenterBlock(head_channels, head_channels);
    else center = torch::nn::Sequential(torch::nn::Identity());
    //the last DecoderBlock of blocks need no skip tensor
    for (int i = 0; i< in_channels.size()-1; i++) {
        blocks->push_back(DecoderBlock(in_channels[i], skip_channels[i], out_channels[i], true, use_attention));
    }
    blocks->push_back(DecoderBlock(in_channels[in_channels.size()-1], skip_channels[in_channels.size()-1],
            out_channels[in_channels.size()-1], false, use_attention));

    register_module("center", center);
    register_module("blocks", blocks);
}

torch::Tensor UNetDecoderImpl::forward(std::vector<torch::Tensor> features){
    std::reverse(std::begin(features),std::end(features));
    torch::Tensor head = features[0];
    features.erase(features.begin());
    auto x = center->forward(head);
    for (int i = 0; i<blocks->size(); i++) {
        x = blocks[i]->as<DecoderBlock>()->forward(x, features[i]);
    }
    return x;
}
