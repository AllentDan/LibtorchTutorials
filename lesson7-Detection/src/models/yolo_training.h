/*
 * This implementation borrows hints from https://github.com/bubbliiiing/yolov4-tiny-pytorch
 * It is acctually a c++ version of yolov4-tiny-pytorch and use weights of its torchscript.
 * Copyright (C) 2021 AllentDan
 * under the MIT license. Writen by AllentDan.
*/
#pragma once
#include <torch/torch.h>
#include <torch/script.h> 

# define Pi 3.14159265358979323846

//索引向量值的位置
template<typename T>
int vec_index(std::vector<T> vec, T value);

//判断是否包含值
template<typename T>
bool in_vec(std::vector<T> vec, T value);


//计算Jaccard系数，tensor shape [:,4]，第二维为center_x, center_y, w, h
torch::Tensor jaccard(torch::Tensor _box_a, torch::Tensor _box_b);

//平滑标签
torch::Tensor smooth_label(torch::Tensor y_true, int label_smoothing, int num_classes);

//	输入为：
//  b1 : tensor, shape = (batch, feat_w, feat_h, anchor_num, 4), xywh
//	b2 : tensor, shape = (batch, feat_w, feat_h, anchor_num, 4), xywh
//	返回为：
//	ciou : tensor, shape = (batch, feat_w, feat_h, anchor_num, 1)
torch::Tensor box_ciou(torch::Tensor b1, torch::Tensor b2);

torch::Tensor clip_by_tensor(torch::Tensor t, float t_min, float t_max);

torch::Tensor MSELoss(torch::Tensor pred, torch::Tensor target);

torch::Tensor BCELoss(torch::Tensor pred, torch::Tensor target);

struct YOLOLossImpl : public torch::nn::Module {
	YOLOLossImpl(torch::Tensor anchors_, int num_classes, int img_size[], float label_smooth = 0, 
		torch::Device device = torch::Device(torch::kCPU), bool normalize = true);
	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<torch::Tensor> targets);
	std::vector<torch::Tensor> get_target(std::vector<torch::Tensor> targets, torch::Tensor scaled_anchors, int in_w, int in_h, float ignore_threshold);
	std::vector<torch::Tensor> get_ignore(torch::Tensor prediction, std::vector<torch::Tensor> targets, torch::Tensor scaled_anchors, int in_w, int in_h, torch::Tensor noobj_mask);
	torch::Tensor anchors;
	int num_anchors = 3;
	int num_classes = 1;
	int bbox_attrs = 0;
	int image_size[2] = { 416,416 };
	float label_smooth = 0;
	std::vector<int> feature_length = { int(image_size[0] / 32),int(image_size[0] / 16),int(image_size[0] / 8) };

	float ignore_threshold = 0.5;
	float lambda_conf = 1.0;
	float lambda_cls = 1.0;
	float lambda_loc = 1.0;
	torch::Device device = torch::Device(torch::kCPU);
	bool normalize = true;
}; //TORCH_MODULE(YOLOLoss);

std::vector< torch::Tensor> non_maximum_suppression(torch::Tensor prediction, int num_classes, float cof_thres = 0.2, float nms_thres = 0.3);

std::vector<int> nms_libtorch(torch::Tensor boxes, torch::Tensor scores, float thresh);