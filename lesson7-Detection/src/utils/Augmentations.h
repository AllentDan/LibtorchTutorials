#pragma once
#include<opencv2/opencv.hpp>

struct  BBox
{
	int xmin = 0;
	int xmax = 0;
	int ymin = 0;
	int ymax = 0;
	std::string name = "";
	int GetH();
	int GetW();
	float CenterX();
	float CenterY();
};

struct Data {
	Data(cv::Mat img, std::vector<BBox> boxes) :image(img), bboxes(boxes) {};
	cv::Mat image;
	std::vector<BBox> bboxes;
};

class Augmentations
{
public:
	static Data Resize(Data mData, int width, int height, float probability);
};

