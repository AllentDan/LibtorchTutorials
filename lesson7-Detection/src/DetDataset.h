#pragma once
#include"utils/util.h"
#include<opencv2/opencv.hpp>
#include"utils/Augmentations.h"


std::vector<BBox> loadXML(std::string xml_path);

class DetDataset :public torch::data::Dataset<DetDataset> {
private:
	// Declare 2 std::vectors of tensors for images and labels
	std::vector<std::string> list_images;
	std::vector<std::string> list_labels;
	bool isTrain = true;
	int width = 416; int height = 416;
	float hflipProb = 0; float vflipProb = 0; float noiseProb = 0; float brightProb = 0;
	float noiseMuLimit = 1; float noiseSigmaLimit = 1; float brightContrastLimit = 0.2; float brightnessLimit = 0;
	std::map<std::string, float> name_idx = {};
public:
	// Constructor
	DetDataset(std::vector<std::string> images, std::vector<std::string> labels, std::vector<std::string> class_names, bool istrain = true,
		int width_ = 416, int height_ = 416, float hflip_prob = 0.5, float vflip_prob = 0)
	{
		list_images = images; list_labels = labels; isTrain = istrain; width = width_; height = height_;
		hflipProb = hflip_prob; vflipProb = vflip_prob;
		for (int i = 0; i < class_names.size(); i++)
		{
			name_idx.insert(std::pair<std::string, float>(class_names[i], float(i)));
		}
	};

	// Override get() function to return tensor at location index
	torch::data::Example<> get(size_t index) override {
		std::string image_path = list_images.at(index);
		std::string annotation_path = list_labels.at(index);

		cv::Mat img = cv::imread(image_path, 1);
		std::vector<BBox> boxes = loadXML(annotation_path);
		Data m_data(img, boxes);

		m_data = Augmentations::Resize(m_data, width, height, 1);
		//Augmentations can be implemented here...

		float width_under1 = 1.0 / m_data.image.cols;//box的长宽归一化用
		float height_under1 = 1.0 / m_data.image.rows;//box的长宽归一化用
		torch::Tensor img_tensor = torch::from_blob(m_data.image.data, { m_data.image.rows, m_data.image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width

		int box_num = m_data.bboxes.size();
		if (m_data.bboxes.size() == 0)
		{
			torch::Tensor label_tensor = torch::ones({ 1 });
			std::cout << annotation_path << std::endl;
			return { img_tensor.clone(), label_tensor.clone() };
		}
		torch::Tensor label_tensor = torch::zeros({ box_num ,5 }).to(torch::kFloat32);
		for (int i = 0; i < box_num; i++)
		{
			label_tensor[i][2] = m_data.bboxes[i].GetW()*width_under1;
			label_tensor[i][3] = m_data.bboxes[i].GetH()*height_under1;
			label_tensor[i][0] = m_data.bboxes[i].xmin*width_under1 + label_tensor[i][2] / 2;
			label_tensor[i][1] = m_data.bboxes[i].ymin*height_under1 + label_tensor[i][3] / 2;
			label_tensor[i][4] = name_idx.at(m_data.bboxes[i].name);
		}
		return { img_tensor.clone(), label_tensor.clone() };
	};

	std::vector<ExampleType> get_batch(c10::ArrayRef<size_t> indices) override {
		std::vector<ExampleType> batch;
		batch.reserve(indices.size());
		for (const auto i : indices) {
			batch.push_back(get(i));
		}
		return batch;
	}

	// Return the length of data
	torch::optional<size_t> size() const override {
		return list_labels.size();
	};
};

//遍历该目录下的.xml文件，并且找到对应的图片
void load_det_data_from_folder(std::string folder, std::string image_type,
	std::vector<std::string> &list_images, std::vector<std::string> &list_labels);