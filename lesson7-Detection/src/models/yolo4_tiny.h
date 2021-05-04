/*
 * This implementation borrows hints from https://github.com/bubbliiiing/yolov4-tiny-pytorch
 * It is acctually a c++ version of yolov4-tiny-pytorch and use weights of its torchscript.
 * Copyright (C) 2021 AllentDan
 * under the MIT license. Writen by AllentDan.
*/
#pragma once
#include"CSPdarknet53_tiny.h"

//conv+upsample
class UpsampleImpl : public torch::nn::Module {
public:
	UpsampleImpl(int in_channels, int out_channels);
	torch::Tensor forward(torch::Tensor x);
private:
	// Declare layers
	torch::nn::Sequential upsample = torch::nn::Sequential();
}; TORCH_MODULE(Upsample);


torch::nn::Sequential yolo_head(std::vector<int> filters_list, int in_filters);

class YoloBody_tinyImpl : public torch::nn::Module {
public:
	YoloBody_tinyImpl(int num_anchors, int num_classes);
	std::vector<torch::Tensor> forward(torch::Tensor x);
private:
	// Declare layers
	CSPdarknet53_tiny backbone{ nullptr };
	BasicConv conv_for_P5{ nullptr };
	Upsample upsample{ nullptr };
	torch::nn::Sequential yolo_headP5{ nullptr };
	torch::nn::Sequential yolo_headP4{ nullptr };
}; TORCH_MODULE(YoloBody_tiny);
