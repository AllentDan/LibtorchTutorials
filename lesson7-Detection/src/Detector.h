#pragma once
#include"utils/util.h"
#include"models/yolo4_tiny.h"
#include<opencv2/opencv.hpp>
class Detector
{
public:
	Detector();
	void Initialize(int gpu_id, int width, int height, std::string name_list_path);
	void Train(std::string train_val_path, std::string image_type, int num_epochs = 30, int batch_size = 4,
		float learning_rate = 0.0003, std::string save_path = "detector.pt", std::string pretrained_path = "detector.pt");
	void LoadWeight(std::string weight_path);
	void loadPretrained(std::string pretrained_pth);
	void Predict(cv::Mat image, bool show = true, float conf_thresh = 0.3, float nms_thresh = 0.3);
private:
	int width = 416; int height = 416; std::vector<std::string> name_list;
	torch::Device device = torch::Device(torch::kCPU);
	YoloBody_tiny detector{ nullptr };
};

