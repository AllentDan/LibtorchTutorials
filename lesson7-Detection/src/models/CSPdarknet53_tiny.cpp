#include "CSPdarknet53_tiny.h"
#include "../utils/util.h"

BasicConvImpl::BasicConvImpl(int in_channels, int out_channels, int kernel_size, 
	int stride) :
	conv(conv_options(in_channels, out_channels, kernel_size, stride, 
		int(kernel_size / 2), 1, false)),
	bn(torch::nn::BatchNorm2d(out_channels)),
	acitivation(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)))
{
	register_module("conv", conv);
	register_module("bn", bn);
}

torch::Tensor BasicConvImpl::forward(torch::Tensor x)
{
	x = conv->forward(x);
	x = bn->forward(x);
	x = acitivation(x);
	return x;
}


Resblock_bodyImpl::Resblock_bodyImpl(int in_channels, int out_channels) {
	this->out_channels = out_channels;
	conv1 = BasicConv(in_channels, out_channels, 3);
	conv2 = BasicConv(out_channels / 2, out_channels / 2, 3);
	conv3 = BasicConv(out_channels / 2, out_channels / 2, 3);
	conv4 = BasicConv(out_channels, out_channels, 1);
	maxpool = torch::nn::MaxPool2d(maxpool_options(2, 2));

	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("conv3", conv3);
	register_module("conv4", conv4);

}
std::vector<torch::Tensor> Resblock_bodyImpl::forward(torch::Tensor x) {
	auto c = out_channels;
	x = conv1->forward(x);
	auto route = x;

	x = torch::split(x, c / 2, 1)[1];
	x = conv2->forward(x);
	auto route1 = x;

	x = conv3->forward(x);
	x = torch::cat({ x, route1 }, 1);
	x = conv4->forward(x);
	auto feat = x;

	x = torch::cat({ route, x }, 1);
	x = maxpool->forward(x);
	return std::vector<torch::Tensor>({ x,feat });
}


CSPdarknet53_tinyImpl::CSPdarknet53_tinyImpl() {
	conv1 = BasicConv(3, 32, 3, 2);
	conv2 = BasicConv(32, 64, 3, 2);
	resblock_body1 = Resblock_body(64, 64);
	resblock_body2 = Resblock_body(128, 128);
	resblock_body3 = Resblock_body(256, 256);
	conv3 = BasicConv(512, 512, 3);

	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("resblock_body1", resblock_body1);
	register_module("resblock_body2", resblock_body2);
	register_module("resblock_body3", resblock_body3);
	register_module("conv3", conv3);
}
std::vector<torch::Tensor> CSPdarknet53_tinyImpl::forward(torch::Tensor x) {
	// 416, 416, 3 -> 208, 208, 32 -> 104, 104, 64
	x = conv1(x);
	x = conv2(x);

	// 104, 104, 64 -> 52, 52, 128
	x = resblock_body1->forward(x)[0];
	// 52, 52, 128 -> 26, 26, 256
	x = resblock_body2->forward(x)[0];
	// 26, 26, 256->xΪ13, 13, 512
#   //        -> feat1Ϊ26,26,256
	auto res_out = resblock_body3->forward(x);
	x = res_out[0];
	auto feat1 = res_out[1];
	// 13, 13, 512 -> 13, 13, 512
	x = conv3->forward(x);
	auto feat2 = x;
	return std::vector<torch::Tensor>({ feat1, feat2 });
}