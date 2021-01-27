#include "vgg.h"

torch::nn::Sequential make_features(std::vector<int> &cfg, bool batch_norm){
    torch::nn::Sequential features;
    int in_channels = 3;
    for(auto v : cfg){
        if(v==-1){
            features->push_back(torch::nn::MaxPool2d(maxpool_options(2,2)));
        }
        else{
            auto conv2d = torch::nn::Conv2d(conv_options(in_channels,v,3,1,1));
            features->push_back(conv2d);
            if(batch_norm){
                features->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(v)));
            }
            features->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
            in_channels = v;
        }
    }
    return features;
}

VGGImpl::VGGImpl(std::vector<int> &cfg, int num_classes, bool batch_norm){
    features_ = make_features(cfg,batch_norm);
    avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));
    classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)));
    classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    classifier->push_back(torch::nn::Dropout());
    classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)));
    classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    classifier->push_back(torch::nn::Dropout());
    classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes)));

    features_ = register_module("features",features_);
    classifier = register_module("classifier",classifier);
}

torch::Tensor VGGImpl::forward(torch::Tensor x){
    x = features_->forward(x);
    x = avgpool(x);
    x = torch::flatten(x,1);
    x = classifier->forward(x);
    return torch::log_softmax(x, 1);
}
