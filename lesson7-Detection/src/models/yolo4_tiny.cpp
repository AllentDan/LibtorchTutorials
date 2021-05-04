#include "yolo4_tiny.h"
#include "../utils/util.h"

UpsampleImpl::UpsampleImpl(int in_channels, int out_channels)
{
	upsample = torch::nn::Sequential(
		BasicConv(in_channels, out_channels, 1)
		//torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2 })).mode(torch::kNearest).align_corners(false))
	);
	register_module("upsample", upsample);
}

torch::Tensor UpsampleImpl::forward(torch::Tensor x)
{
	x = upsample->forward(x);
	x = at::upsample_nearest2d(x, { x.sizes()[2] * 2 , x.sizes()[3] * 2 });
	return x;
}

torch::nn::Sequential yolo_head(std::vector<int> filters_list, int in_filters) {
	auto m = torch::nn::Sequential(BasicConv(in_filters, filters_list[0], 3),
		torch::nn::Conv2d(conv_options(filters_list[0], filters_list[1], 1)));
	return m;
}

YoloBody_tinyImpl::YoloBody_tinyImpl(int num_anchors, int num_classes) {
	backbone = CSPdarknet53_tiny();
	conv_for_P5 = BasicConv(512, 256, 1);
	yolo_headP5 = yolo_head({ 512, num_anchors * (5 + num_classes) }, 256);
	upsample = Upsample(256, 128);
	yolo_headP4 = yolo_head({ 256, num_anchors * (5 + num_classes) }, 384);

	register_module("backbone", backbone);
	register_module("conv_for_P5", conv_for_P5);
	register_module("yolo_headP5", yolo_headP5);
	register_module("upsample", upsample);
	register_module("yolo_headP4", yolo_headP4);
}
std::vector<torch::Tensor> YoloBody_tinyImpl::forward(torch::Tensor x) {
	//return feat1 with shape of {26,26,256} and feat2 of {13, 13, 512}
	auto backbone_out = backbone->forward(x);
	auto feat1 = backbone_out[0];
	auto feat2 = backbone_out[1];
	//13,13,512 -> 13,13,256
	auto P5 = conv_for_P5->forward(feat2);
	//13, 13, 256 -> 13, 13, 512 -> 13, 13, 255
	auto out0 = yolo_headP5->forward(P5);


	//13,13,256 -> 13,13,128 -> 26,26,128
	auto P5_Upsample = upsample->forward(P5);
	//26, 26, 256 + 26, 26, 128 -> 26, 26, 384
	auto P4 = torch::cat({ P5_Upsample, feat1 }, 1);
	//26, 26, 384 -> 26, 26, 256 -> 26, 26, 255
	auto out1 = yolo_headP4->forward(P4);
	return std::vector<torch::Tensor>({ out0, out1 });
}