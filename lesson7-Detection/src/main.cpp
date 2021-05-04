#include<opencv2/opencv.hpp>
#include"models/yolo4_tiny.h"
#include"models/yolo_training.h"
#include"Detector.h"

void show_bbox_coco(cv::Mat image, torch::Tensor bboxes, int nums) {
	//设置绘制文本的相关参数
	int font_face = cv::FONT_HERSHEY_COMPLEX;
	double font_scale = 0.4;
	int thickness = 1;
	float* bbox = new float[bboxes.size(0)]();
	std::cout << bboxes << std::endl;
	memcpy(bbox, bboxes.cpu().data_ptr(), bboxes.size(0) * sizeof(float));
	for (int i = 0; i < bboxes.size(0); i = i + 7)
	{
		cv::rectangle(image, cv::Rect(bbox[i + 0], bbox[i + 1], bbox[i + 2] - bbox[i + 0], bbox[i + 3] - bbox[i + 1]), cv::Scalar(0, 0, 255));
		//将文本框居中绘制
		cv::Point origin;
		origin.x = bbox[i + 0];
		origin.y = bbox[i + 1] + 8;
		cv::putText(image, std::to_string(int(bbox[i + 6])), origin, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 1, 0);
	}
	delete bbox;
	cv::imwrite("prediction_coco.jpg", image);
	cv::imshow("test", image);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

void Predict(YoloBody_tiny detector, cv::Mat image, bool show, float conf_thresh, float nms_thresh) {
	int origin_width = image.cols;
	int origin_height = image.rows;
	cv::resize(image, image, { 416,416 });
	auto img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat) / 255.0;

	float anchor[12] = { 10,14,  23,27,  37,58,  81,82,  135,169,  344,319 };
	auto anchors_ = torch::from_blob(anchor, { 6,2 }, torch::TensorOptions(torch::kFloat32));
	int image_size[2] = { 416,416 };
	img_tensor = img_tensor.cuda();

	auto outputs = detector->forward(img_tensor);
	std::vector<torch::Tensor> output_list = {};
	auto tensor_input = outputs[1];
	auto output_decoded = DecodeBox(tensor_input, anchors_.narrow(0, 0, 3), 80, image_size);
	output_list.push_back(output_decoded);

	tensor_input = outputs[0];
	output_decoded = DecodeBox(tensor_input, anchors_.narrow(0, 3, 3), 80, image_size);
	output_list.push_back(output_decoded);

	//std::cout << tensor_input << anchors_.narrow(0, 3, 3);

	auto output = torch::cat(output_list, 1);
	auto detection = non_maximum_suppression(output, 80, conf_thresh, nms_thresh);

	float w_scale = float(origin_width) / 416;
	float h_scale = float(origin_height) / 416;
	for (int i = 0; i < detection.size(); i++) {
		for (int j = 0; j < detection[i].size(0) / 7; j++)
		{
			detection[i].select(0, 7 * j + 0) *= w_scale;
			detection[i].select(0, 7 * j + 1) *= h_scale;
			detection[i].select(0, 7 * j + 2) *= w_scale;
			detection[i].select(0, 7 * j + 3) *= h_scale;
		}
	}

	cv::resize(image, image, { origin_width,origin_height });
	if (show)
		show_bbox_coco(image, detection[0], 80);
	return;
}

int main()
{
	//float tt[20] = { 0.4928,  0.2981,  0.1731,  0.4038, 17.0000,0.3317,  0.7163,  0.5673,  0.2548,  0.0000,0.8714,
	//0.8125,  0.1370,  0.2308, 14.0000,0.6250,  0.6202,  0.0144,  0.0240, 14.0000 };
	//auto targets = torch::from_blob(tt, { 4,5 }, torch::TensorOptions(torch::kFloat32));

	//float anchor[12] = { 10,14,  23,27,  37,58,  81,82,  135,169,  344,319 };
	//auto anchors_ = torch::from_blob(anchor, { 6,2 }, torch::TensorOptions(torch::kFloat32)).to(torch::Device(torch::kCUDA));
	//int image_size[2] = { 416, 416 };

	//bool normalize = false;
	//auto critia1 = YOLOLossImpl(anchors_, 20, image_size, 0.01, torch::Device(torch::kCUDA), normalize);

	//auto output = torch::ones({ 1,75,13,13 }, torch::kFloat).to(torch::Device(torch::kCUDA));
	////auto tmp = torch::tensor({ 0.2, 0.2, 0.3, 0.3, 1.0 }, torch::kFloat).unsqueeze(0).to(torch::Device(torch::kCUDA));
	////std::cout << tmp.sizes();
	//std::vector<torch::Tensor> targets_vec = { targets };
	//auto loss_numpos1 = critia1.forward(output, targets_vec);
	//std::cout << loss_numpos1[0];



	cv::Mat image = cv::imread("D:\\AllentFiles\\data\\dataset4work\\detection\\val\\images\\2007_005331.jpg");
	Detector detector;
	detector.Initialize(-1, 416, 416, "D:\\AllentFiles\\data\\dataset4work\\detection\\name.txt");
	//detector.Train("D:\\AllentFiles\\data\\dataset4work\\detection", ".jpg", 30,
	//	4, 0.001, "weights/detector.pt", "weights/yolo4_tiny.pt");

	detector.LoadWeight("weights/detector.pt");


	auto model = YoloBody_tiny(3, 80);
	torch::load(model, "weights/yolo4_tiny.pt");
	model->to(torch::kCUDA);
	Predict(model, image, true, 0.1, 0.3);

	detector.Predict(image, true, 0.1);
	int64 start = cv::getTickCount();
	int loops = 10;
	for (int i = 0; i < loops; i++) {
		detector.Predict(image, false);
	}
	double duration = (cv::getTickCount() - start) / cv::getTickFrequency();
	std::cout << duration/ loops <<" s per prediction" << std::endl;

	return 0;
}