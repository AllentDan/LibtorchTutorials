#include "yolo_training.h"

template<typename T>
int vec_index(std::vector<T> vec, T value) {
	int a = 0;
	for (auto temp : vec)
	{
		if (temp == value)
		{
			return a;
		}
		a++;
	}
	if (a == vec.size())
	{
		std::cout << "No such value in std::vector" << std::endl;
		return a;
	}
}

//索引向量值的位置
template<typename T>
bool in_vec(std::vector<T> vec, T value) {
	for (auto temp : vec)
	{
		if (temp == value)
		{
			return true;
		}
	}
	return false;
}


torch::Tensor jaccard(torch::Tensor _box_a, torch::Tensor _box_b) {
	//auto TensorType = _box_b.options();
	auto b1_x1 = _box_a.select(1, 0) - _box_a.select(1, 2) / 2;
	auto b1_x2 = _box_a.select(1, 0) + _box_a.select(1, 2) / 2;
	auto b1_y1 = _box_a.select(1, 1) - _box_a.select(1, 3) / 2;
	auto b1_y2 = _box_a.select(1, 1) + _box_a.select(1, 3) / 2;

	auto b2_x1 = _box_b.select(1, 0) - _box_b.select(1, 2) / 2;
	auto b2_x2 = _box_b.select(1, 0) + _box_b.select(1, 2) / 2;
	auto b2_y1 = _box_b.select(1, 1) - _box_b.select(1, 3) / 2;
	auto b2_y2 = _box_b.select(1, 1) + _box_b.select(1, 3) / 2;


	auto box_a = torch::zeros_like(_box_a);
	auto box_b = torch::zeros_like(_box_b);

	box_a.select(1, 0) = b1_x1;
	box_a.select(1, 1) = b1_y1;
	box_a.select(1, 2) = b1_x2;
	box_a.select(1, 3) = b1_y2;

	box_b.select(1, 0) = b2_x1;
	box_b.select(1, 1) = b2_y1;
	box_b.select(1, 2) = b2_x2;
	box_b.select(1, 3) = b2_y2;

	auto A = box_a.size(0);
	auto B = box_b.size(0);

	//try
	//{
	//	auto max_xy = torch::min(box_a.narrow(1, 2, 2).unsqueeze(1).expand({ A, B, 2 }), box_b.narrow(1, 2, 2).unsqueeze(0).expand({ A, B, 2 }));
	//	auto min_xy = torch::max(box_a.narrow(1, 0, 2).unsqueeze(1).expand({ A, B, 2 }), box_b.narrow(1, 0, 2).unsqueeze(0).expand({ A, B, 2 }));
	//}
	//catch (const std::exception&e)
	//{
	//	cout << e.what() << endl;
	//}
	auto max_xy = torch::min(box_a.narrow(1, 2, 2).unsqueeze(1).expand({ A, B, 2 }), box_b.narrow(1, 2, 2).unsqueeze(0).expand({ A, B, 2 }));
	auto min_xy = torch::max(box_a.narrow(1, 0, 2).unsqueeze(1).expand({ A, B, 2 }), box_b.narrow(1, 0, 2).unsqueeze(0).expand({ A, B, 2 }));

	auto inter = torch::clamp((max_xy - min_xy), 0);
	inter = inter.select(2, 0) * inter.select(2, 1);

	//计算先验框和真实框各自的面积
	auto area_a = ((box_a.select(1, 2) - box_a.select(1, 0)) * (box_a.select(1, 3) - box_a.select(1, 1))).unsqueeze(1).expand_as(inter); // [A, B]
	auto area_b = ((box_b.select(1, 2) - box_b.select(1, 0)) * (box_b.select(1, 3) - box_b.select(1, 1))).unsqueeze(0).expand_as(inter);  // [A, B]

	//求IOU
	auto uni = area_a + area_b - inter;
	return inter / uni;  // [A, B]
}


torch::Tensor smooth_label(torch::Tensor y_true, int label_smoothing, int num_classes) {
	return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes;
}

torch::Tensor box_ciou(torch::Tensor b1, torch::Tensor b2)
{
	//求出预测框左上角右下角
	auto b1_xy = b1.narrow(-1, 0, 2);
	auto b1_wh = b1.narrow(-1, 2, 2);
	auto b1_wh_half = b1_wh / 2.0;
	auto b1_mins = b1_xy - b1_wh_half;
	auto b1_maxes = b1_xy + b1_wh_half;

	//求出真实框左上角右下角
	auto b2_xy = b2.narrow(-1, 0, 2);
	auto b2_wh = b2.narrow(-1, 2, 2);
	auto b2_wh_half = b2_wh / 2.0;
	auto b2_mins = b2_xy - b2_wh_half;
	auto b2_maxes = b2_xy + b2_wh_half;

	// 求真实框和预测框所有的iou
	auto intersect_mins = torch::max(b1_mins, b2_mins);
	auto intersect_maxes = torch::min(b1_maxes, b2_maxes);
	auto intersect_wh = torch::max(intersect_maxes - intersect_mins, torch::zeros_like(intersect_maxes));
	auto intersect_area = intersect_wh.select(-1, 0) * intersect_wh.select(-1, 1);
	auto b1_area = b1_wh.select(-1, 0) * b1_wh.select(-1, 1);
	auto b2_area = b2_wh.select(-1, 0) * b2_wh.select(-1, 1);
	auto union_area = b1_area + b2_area - intersect_area;
	auto iou = intersect_area / torch::clamp(union_area, 1e-6);

	//计算中心的差距
	auto center_distance = torch::sum(torch::pow((b1_xy - b2_xy), 2), -1);

	//找到包裹两个框的最小框的左上角和右下角
	auto enclose_mins = torch::min(b1_mins, b2_mins);
	auto enclose_maxes = torch::max(b1_maxes, b2_maxes);
	auto enclose_wh = torch::max(enclose_maxes - enclose_mins, torch::zeros_like(intersect_maxes));

	//计算对角线距离
	auto enclose_diagonal = torch::sum(torch::pow(enclose_wh, 2), -1);
	auto ciou = iou - 1.0 * (center_distance) / (enclose_diagonal + 1e-7);

	auto v = (4 / (Pi * Pi)) * torch::pow((torch::atan(b1_wh.select(-1,0) / b1_wh.select(-1,1)) - torch::atan(b2_wh.select(-1,0) / b2_wh.select(-1,1))), 2);
	auto alpha = v / (1.0 - iou + v);
	ciou = ciou - alpha * v;

	return ciou;
}

//clip tensor, 类型是32还是64存疑，后改
torch::Tensor clip_by_tensor(torch::Tensor t, float t_min, float t_max) {
	t = t.to(torch::kFloat32);
	auto result = (t >= t_min).to(torch::kFloat32) * t + (t < t_min).to(torch::kFloat32) * t_min;
	result = (result <= t_max).to(torch::kFloat32) * result + (result > t_max).to(torch::kFloat32) * t_max;
	return result;
}

torch::Tensor MSELoss(torch::Tensor pred, torch::Tensor target) {
	return torch::pow((pred - target), 2);
}

torch::Tensor BCELoss(torch::Tensor pred, torch::Tensor target) {
	pred = clip_by_tensor(pred, 1e-7, 1.0 - 1e-7);
	auto output = -target * torch::log(pred) - (1.0 - target) * torch::log(1.0 - pred);
	return output;
}


YOLOLossImpl::YOLOLossImpl(torch::Tensor anchors_, int num_classes_, int img_size_[], 
	float label_smooth_, torch::Device device_, bool normalize) {
	this->anchors = anchors_;
	this->num_anchors = anchors_.sizes()[0];
	this->num_classes = num_classes_;
	this->bbox_attrs = 5 + num_classes;
	memcpy(image_size, img_size_, 2 * sizeof(int));
	std::vector<int> feature_length_ = { int(image_size[0] / 32),int(image_size[0] / 16),int(image_size[0] / 8) };
	std::copy(feature_length_.begin(), feature_length_.end(), feature_length.begin());
	this->label_smooth = label_smooth_;
	this->device = device_;
	this->normalize = normalize;
}

std::vector<torch::Tensor> YOLOLossImpl::forward(torch::Tensor input, std::vector<torch::Tensor> targets)
{
	//input为bs, 3 * (5 + num_classes), 13, 13
	//一共多少张图片
	auto bs = input.size(0);
	//特征层的高
	auto in_h = input.size(2);
	//特征层的宽
	auto in_w = input.size(3);

	//计算步长，每一个特征点对应原来的图片上多少个像素点
	//如果特征层为13x13的话，原图416x416的情况下，一个特征点就对应原来的图片上的32个像素点
	auto stride_h = image_size[1] / in_h;
	auto stride_w = image_size[0] / in_w;

	//把先验框的尺寸调整成特征层大小的形式
	//计算出先验框在特征层上对应的宽高
	auto scaled_anchors = anchors.clone();
	scaled_anchors.select(1, 0) = scaled_anchors.select(1, 0) / stride_w;
	scaled_anchors.select(1, 1) = scaled_anchors.select(1, 1) / stride_h;

	//bs, 3 * (5 + num_classes), 13, 13->bs, 3, 13, 13, (5 + num_classes)
	auto prediction = input.view({bs, int(num_anchors / 2), bbox_attrs, in_h, in_w}).permute({0, 1, 3, 4, 2}).contiguous();
	//对prediction预测进行调整
	auto conf = torch::sigmoid(prediction.select(-1,4));//  # Conf
	auto pred_cls = torch::sigmoid(prediction.narrow(-1, 5, num_classes));  //Cls pred.

	//找到哪些先验框内部包含物体
	auto temp = get_target(targets, scaled_anchors, in_w, in_h, ignore_threshold);
	auto BoolType = torch::ones(1).to(torch::kBool).to(device).options();
	auto FloatType = torch::ones(1).to(torch::kFloat).to(device).options();
	auto mask = temp[0].to(BoolType); 
	auto noobj_mask = temp[1].to(device);
	auto t_box = temp[2];  
	auto tconf = temp[3];  
	auto tcls = temp[4];  
	auto box_loss_scale_x = temp[5];
	auto box_loss_scale_y = temp[6];

	auto temp_ciou = get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask);
	noobj_mask = temp_ciou[0];
	auto pred_boxes_for_ciou = temp_ciou[1];


	mask = mask.to(device);
	noobj_mask = noobj_mask.to(device);
	box_loss_scale_x = box_loss_scale_x.to(device);
	box_loss_scale_y =  box_loss_scale_y.to(device);
	tconf = tconf.to(device);
	tcls = tcls.to(device);
	pred_boxes_for_ciou = pred_boxes_for_ciou.to(device);
	t_box = t_box.to(device);


	auto box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y;
	auto ciou = (1 - box_ciou(pred_boxes_for_ciou.index({mask}), t_box.index({ mask })))* box_loss_scale.index({ mask });
	auto loss_loc = torch::sum(ciou / bs);

	auto loss_conf = torch::sum(BCELoss(conf, mask.to(FloatType)) * mask.to(FloatType) / bs) + \
		torch::sum(BCELoss(conf, mask.to(FloatType)) * noobj_mask / bs);

	auto loss_cls = torch::sum(BCELoss(pred_cls.index({ mask==1 }), smooth_label(tcls.index({ mask==1 }), label_smooth, num_classes)) / bs);
	auto loss = loss_conf * lambda_conf + loss_cls * lambda_cls + loss_loc * lambda_loc;

	//std::cout << mask.sum();
	//std::cout << loss.item()<< std::endl<< loss_conf<< loss_cls<< loss_loc << std::endl;
	torch::Tensor num_pos = torch::tensor({0}).to(device);
	if (normalize) {
		num_pos = torch::sum(mask);
		num_pos = torch::max(num_pos, torch::ones_like(num_pos));
	}
	else
		num_pos[0] = bs / 2;
	return std::vector<torch::Tensor>({ loss, num_pos });
}

std::vector<torch::Tensor> YOLOLossImpl::get_target(std::vector<torch::Tensor> targets, torch::Tensor scaled_anchors, int in_w, int in_h, float ignore_threshold)
{
	//计算一共有多少张图片
	int bs = targets.size();
	auto scaled_anchorsType = scaled_anchors.options();
	//获得先验框
	int index = vec_index(feature_length, in_w);
	std::vector<std::vector<int>> anchor_vec_in_vec = { {3, 4, 5} ,{0, 1, 2}};
	std::vector<int> anchor_index = anchor_vec_in_vec[index];
	int subtract_index = 3*index;//0或者3或者6
	//创建全是0或者全是1的阵列
	torch::TensorOptions grad_false(torch::requires_grad(false));
	auto TensorType = targets[0].options();
	auto mask = torch::zeros({ bs, int(num_anchors / 2), in_h, in_w }, grad_false);
	auto noobj_mask = torch::ones({ bs, int(num_anchors / 2), in_h, in_w }, grad_false);

	auto tx = torch::zeros({bs, int(num_anchors / 2), in_h, in_w}, grad_false);
	auto ty = torch::zeros({bs, int(num_anchors / 2), in_h, in_w}, grad_false);
	auto tw = torch::zeros({bs, int(num_anchors / 2), in_h, in_w}, grad_false);
	auto th = torch::zeros({bs, int(num_anchors / 2), in_h, in_w}, grad_false);
	auto t_box = torch::zeros({bs, int(num_anchors / 2), in_h, in_w, 4}, grad_false);
	auto tconf = torch::zeros({bs, int(num_anchors / 2), in_h, in_w}, grad_false);
	auto tcls = torch::zeros({bs, int(num_anchors / 2), in_h, in_w, num_classes}, grad_false);

	auto box_loss_scale_x = torch::zeros({bs, int(num_anchors / 2), in_h, in_w}, grad_false);
	auto box_loss_scale_y = torch::zeros({bs, int(num_anchors / 2), in_h, in_w}, grad_false);
	for (int b = 0; b < bs; b++)
	{
		if (targets[b].sizes().size() == 1)//dataset设置targets中图片无box则tensor = torch::ones({1});
		{
			continue;
		}
		//计算出在特征层上的点位
		auto gxs = targets[b].narrow(-1,0,1) * in_w;
		auto gys = targets[b].narrow(-1, 1, 1) * in_h;

		auto gws = targets[b].narrow(-1, 2, 1) * in_w;
		auto ghs = targets[b].narrow(-1, 3, 1) * in_h;

		//计算出属于哪个网格
		auto gis = torch::floor(gxs);
		auto gjs = torch::floor(gys);

		//计算真实框的位置
		auto gt_box = torch::Tensor(torch::cat({ torch::zeros_like(gws), torch::zeros_like(ghs), gws, ghs }, 1)).to(torch::kFloat32);

		//计算出所有先验框的位置
		auto anchor_shapes = torch::Tensor(torch::cat({ torch::zeros({ num_anchors, 2 }).to(scaled_anchorsType), torch::Tensor(scaled_anchors) }, 1)).to(TensorType);
		//计算重合程度
		auto anch_ious = jaccard(gt_box, anchor_shapes);

		//Find the best matching anchor box
		auto best_ns = torch::argmax(anch_ious, -1);

		for (int i = 0; i < best_ns.sizes()[0]; i++)
		{
			if (!in_vec(anchor_index, best_ns[i].item().toInt()))
			{
				continue;
			}
			auto gi = gis[i].to(torch::kLong).item().toInt();
			auto gj = gjs[i].to(torch::kLong).item().toInt();
			auto gx = gxs[i].item().toFloat();
			auto gy = gys[i].item().toFloat();
			auto gw = gws[i].item().toFloat();
			auto gh = ghs[i].item().toFloat();
			if (gj < in_h && gi < in_w) {
				auto best_n = vec_index(anchor_index, best_ns[i].item().toInt());// (best_ns[i] - subtract_index).item().toInt();
					//判定哪些先验框内部真实的存在物体

				noobj_mask[b][best_n][gj][gi] = 0;
				mask[b][best_n][gj][gi] = 1;
				//计算先验框中心调整参数
				tx[b][best_n][gj][gi] = gx;
				ty[b][best_n][gj][gi] = gy;
				//计算先验框宽高调整参数
				tw[b][best_n][gj][gi] = gw;
				th[b][best_n][gj][gi] = gh;
				//用于获得xywh的比例
				box_loss_scale_x[b][best_n][gj][gi] = targets[b][i][2];
				box_loss_scale_y[b][best_n][gj][gi] = targets[b][i][3];
				//物体置信度
				tconf[b][best_n][gj][gi] = 1;
				//种类
				tcls[b][best_n][gj][gi][targets[b][i][4].item().toLong()] = 1;
			}
			else {
				std::cout << gxs << gys << std::endl;
				std::cout << gis << gjs << std::endl;
				std::cout << targets[b];
				std::cout << "Step out of boundary;" << std::endl;
				//cout << targets[i] << endl;
			}

		}
	}
	t_box.select(-1, 0) = tx;
	t_box.select(-1, 1) = ty;
	t_box.select(-1, 2) = tw;
	t_box.select(-1, 3) = th;
	std::vector<torch::Tensor> output = { mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y };
	return output;
}

std::vector<torch::Tensor> YOLOLossImpl::get_ignore(torch::Tensor prediction, std::vector<torch::Tensor> targets, torch::Tensor scaled_anchors, int in_w, int in_h, torch::Tensor noobj_mask)
{
	int bs = targets.size();
	int index = vec_index(feature_length, in_w);
	std::vector<std::vector<int>> anchor_vec_in_vec = { {3, 4, 5}, {0, 1, 2}};
	std::vector<int> anchor_index = anchor_vec_in_vec[index];
	//先验框的中心位置的调整参数
	auto x = torch::sigmoid(prediction.select(-1,0));
	auto y = torch::sigmoid(prediction.select(-1,1));
	//先验框的宽高调整参数
	auto w = prediction.select(-1,2);  //Width
	auto h = prediction.select(-1,3);  // Height


	auto FloatType = prediction.options();
	auto LongType = prediction.to(torch::kLong).options();
	
	//生成网格，先验框中心，网格左上角
	auto grid_x = torch::linspace(0, in_w - 1, in_w).repeat({in_h, 1}).repeat(
		{int(bs*num_anchors / 2), 1, 1}).view(x.sizes()).to(FloatType);
	auto grid_y = torch::linspace(0, in_h - 1, in_h).repeat({in_w, 1}).t().repeat(
		{int(bs*num_anchors / 2), 1, 1}).view(y.sizes()).to(FloatType);

	auto anchor_w = scaled_anchors.narrow(0, anchor_index[0], 3).narrow(-1, 0, 1).to(FloatType);
	auto anchor_h = scaled_anchors.narrow(0, anchor_index[0], 3).narrow(-1, 1, 1).to(FloatType);
	anchor_w = anchor_w.repeat({bs, 1}).repeat({1, 1, in_h * in_w}).view(w.sizes());
	anchor_h = anchor_h.repeat({bs, 1}).repeat({1, 1, in_h * in_w}).view(h.sizes());
	
	//计算调整后的先验框中心与宽高
	auto pred_boxes = torch::randn_like(prediction.narrow(-1, 0, 4)).to(FloatType);
	pred_boxes.select(-1, 0) = x + grid_x;
	pred_boxes.select(-1, 1) = y + grid_y;
	
	pred_boxes.select(-1, 2) = w.exp() * anchor_w;
	pred_boxes.select(-1, 3) = h.exp() * anchor_h;

	for (int i=0; i<bs;i++)
	{
		auto pred_boxes_for_ignore = pred_boxes[i];
		pred_boxes_for_ignore = pred_boxes_for_ignore.view({-1, 4});
		if (targets[i].sizes().size() >1) {
			auto gx = targets[i].narrow(-1,0,1) * in_w;
			auto gy = targets[i].narrow(-1, 1, 1) * in_h;
			auto gw = targets[i].narrow(-1, 2, 1) * in_w;
			auto gh = targets[i].narrow(-1, 3, 1) * in_h;
			auto gt_box = torch::cat({ gx, gy, gw, gh }, -1).to(FloatType);

			auto anch_ious = jaccard(gt_box, pred_boxes_for_ignore);
			auto anch_ious_max_tuple = torch::max(anch_ious, 0);
			auto anch_ious_max = std::get<0>(anch_ious_max_tuple);

			anch_ious_max = anch_ious_max.view(pred_boxes.sizes().slice(1, 3));
			noobj_mask[i] = (anch_ious_max <= ignore_threshold).to(FloatType)*noobj_mask[i];
		}

	}

	std::vector<torch::Tensor> output = {noobj_mask, pred_boxes};
	return output;
}

std::vector<torch::Tensor> non_maximum_suppression(torch::Tensor prediction, int num_classes, float conf_thres, float nms_thres) {

	prediction.select(-1, 0) -= prediction.select(-1, 2) / 2;
	prediction.select(-1, 1) -= prediction.select(-1, 3) / 2;
	prediction.select(-1, 2) += prediction.select(-1, 0);
	prediction.select(-1, 3) += prediction.select(-1, 1);

	std::vector<torch::Tensor> output;
	for (int image_id = 0; image_id < prediction.sizes()[0]; image_id++) {
		auto image_pred = prediction[image_id];
		auto max_out_tuple = torch::max(image_pred.narrow(-1, 5, num_classes), -1, true);
		auto class_conf = std::get<0>(max_out_tuple);
		auto class_pred = std::get<1>(max_out_tuple);
		auto conf_mask = (image_pred.select(-1, 4) * class_conf.select(-1, 0) >= conf_thres).squeeze();
		image_pred = image_pred.index({ conf_mask }).to(torch::kFloat);
		class_conf = class_conf.index({ conf_mask }).to(torch::kFloat);
		class_pred = class_pred.index({ conf_mask }).to(torch::kFloat);

		if (!image_pred.sizes()[0]) {
			output.push_back(torch::full({ 1, 7 }, 0));
			continue;
		}

		//获得的内容为(x1, y1, x2, y2, obj_conf, class_conf, class_pred)
		auto detections = torch::cat({ image_pred.narrow(-1,0,5), class_conf, class_pred }, 1);
		//获得种类
		std::vector<torch::Tensor> img_classes;

		for (int m = 0, len = detections.size(0); m < len; m++)
		{
			bool found = false;
			for (size_t n = 0; n < img_classes.size(); n++)
			{
				auto ret = (detections[m][6] == img_classes[n]);
				if (torch::nonzero(ret).size(0) > 0)
				{
					found = true;
					break;
				}
			}
			if (!found) img_classes.push_back(detections[m][6]);
		}
		std::vector<torch::Tensor> temp_class_detections;
		for (auto c : img_classes) {
			auto detections_class = detections.index({ detections.select(-1,-1) == c });
			auto keep = nms_libtorch(detections_class.narrow(-1, 0, 4), detections_class.select(-1, 4)*detections_class.select(-1, 5), nms_thres);
			std::vector<torch::Tensor> temp_max_detections;
			for (auto v : keep) {
				temp_max_detections.push_back(detections_class[v]);
			}
			auto max_detections = torch::cat(temp_max_detections, 0);
			temp_class_detections.push_back(max_detections);
		}
		auto class_detections = torch::cat(temp_class_detections, 0);
		output.push_back(class_detections);
	}
	return output;
}

std::vector<int> nms_libtorch(torch::Tensor bboxes, torch::Tensor scores, float thresh) {
	auto x1 = bboxes.select(-1, 0);
	auto y1 = bboxes.select(-1, 1);
	auto x2 = bboxes.select(-1, 2);
	auto y2 = bboxes.select(-1, 3);
	auto areas = (x2 - x1)*(y2 - y1);   //[N, ] 每个bbox的面积
	auto tuple_sorted = scores.sort(0, true);    //降序排列
	auto order = std::get<1>(tuple_sorted);

	std::vector<int>	keep;
	while (order.numel() > 0) {// torch.numel()返回张量元素个数
		if (order.numel() == 1) {//    保留框只剩一个
			auto i = order.item();
			keep.push_back(i.toInt());
			break;
		}
		else {
			auto i = order[0].item();// 保留scores最大的那个框box[i]
			keep.push_back(i.toInt());
		}
		//计算box[i]与其余各框的IOU(思路很好)
		auto order_mask = order.narrow(0, 1, order.size(-1) - 1);
		x1.index({ order_mask });
		x1.index({ order_mask }).clamp(x1[keep.back()].item().toFloat(), 1e10);
		auto xx1 = x1.index({ order_mask }).clamp(x1[keep.back()].item().toFloat(), 1e10);// [N - 1, ]
		auto yy1 = y1.index({ order_mask }).clamp(y1[keep.back()].item().toFloat(), 1e10);
		auto xx2 = x2.index({ order_mask }).clamp(0, x2[keep.back()].item().toFloat());
		auto yy2 = y2.index({ order_mask }).clamp(0, y2[keep.back()].item().toFloat());
		auto inter = (xx2 - xx1).clamp(0, 1e10) * (yy2 - yy1).clamp(0, 1e10);// [N - 1, ]

		auto iou = inter / (areas[keep.back()] + areas.index({ order.narrow(0,1,order.size(-1) - 1) }) - inter);//[N - 1, ]
		auto idx = (iou <= thresh).nonzero().squeeze();//注意此时idx为[N - 1, ] 而order为[N, ]
		if (idx.numel() == 0) {
			break;
		}
		order = order.index({ idx + 1 }); //修补索引之间的差值
	}
	return keep;
}

