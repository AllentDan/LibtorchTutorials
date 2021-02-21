#ifndef SEGDATASET_H
#define SEGDATASET_H
#include"util.h"
#include"fstream"
#include "json.hpp"
#include<opencv2/opencv.hpp>

void show_mask(std::string json_path, std::string image_type = ".jpg");
void draw_mask(std::string json_path, cv::Mat &mask);

class SegDataset :public torch::data::Dataset<SegDataset>
{
public:
    SegDataset(int resize_width, int resize_height, std::vector<std::string> list_images,
               std::vector<std::string> list_labels, std::vector<std::string> name_list);
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override;
    // Return the length of data
    torch::optional<size_t> size() const override {
        return list_labels.size();
    };
private:
    void draw_mask(std::string json_path, cv::Mat &mask);
    int resize_width = 512; int resize_height = 512;
    std::vector<std::string> name_list = {};
    std::map<std::string, int> name2index = {};
    std::map<std::string, cv::Scalar> name2color = {};
    std::vector<std::string> list_images;
    std::vector<std::string> list_labels;
};

#endif // SEGDATASET_H
