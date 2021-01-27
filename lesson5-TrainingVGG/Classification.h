#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H
#include<vgg.h>
#include<dataSet.h>
#include<opencv2/opencv.hpp>

class Classifier
{
private:
    torch::Device device = torch::Device(torch::kCPU);
    VGG vgg = VGG{nullptr};
public:
    Classifier(int gpu_id = 0);
    void Initialize(int num_classes, std::string pretrained_path);
    void Train(int epochs, int batch_size, float learning_rate, std::string train_val_dir, std::string image_type, std::string save_path);
    int Predict(cv::Mat &image);
    void LoadWeight(std::string weight);
};

#endif // CLASSIFICATION_H
