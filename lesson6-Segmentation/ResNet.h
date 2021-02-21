#ifndef RESNET_H
#define RESNET_H
#include"util.h"

class BlockImpl : public torch::nn::Module {
public:
    BlockImpl(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
        torch::nn::Sequential downsample_ = nullptr, int groups = 1, int base_width = 64, bool is_basic = true);
    torch::Tensor forward(torch::Tensor x);
    torch::nn::Sequential downsample{ nullptr };
private:
    bool is_basic = true;
    int64_t stride = 1;
    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::BatchNorm2d bn2{ nullptr };
    torch::nn::Conv2d conv3{ nullptr };
    torch::nn::BatchNorm2d bn3{ nullptr };

};
TORCH_MODULE(Block);


class ResNetImpl : public torch::nn::Module {
public:
    ResNetImpl(std::vector<int> layers, int num_classes = 1000, std::string model_type = "resnet18",
		int groups = 1, int width_per_group = 64);
    torch::Tensor forward(torch::Tensor x);
    std::vector<torch::Tensor> features(torch::Tensor x);
    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1);
	void make_dilated(std::vector<int> stage_list, std::vector<int> dilation_list);
private:
    int expansion = 1; bool is_basic = true;
	int64_t inplanes = 64; int groups = 1; int base_width = 64;
    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr };
    torch::nn::Sequential layer1{ nullptr };
    torch::nn::Sequential layer2{ nullptr };
    torch::nn::Sequential layer3{ nullptr };
    torch::nn::Sequential layer4{ nullptr };
    torch::nn::Linear fc{nullptr};
};
TORCH_MODULE(ResNet);

inline std::map<std::string, std::vector<int>> getParams(){
    std::map<std::string, std::vector<int>> name2layers = {};
    name2layers.insert(std::pair<std::string, std::vector<int>>("resnet18",{2, 2, 2, 2}));
    name2layers.insert(std::pair<std::string, std::vector<int>>("resnet34",{3, 4, 6, 3}));
    name2layers.insert(std::pair<std::string, std::vector<int>>("resnet50",{3, 4, 6, 3}));
    name2layers.insert(std::pair<std::string, std::vector<int>>("resnet101",{3, 4, 23, 3}));
	name2layers.insert(std::pair<std::string, std::vector<int>>("resnet152", { 3, 8, 36, 3 }));
	name2layers.insert(std::pair<std::string, std::vector<int>>("resnext50_32x4d", { 3, 4, 6, 3 }));
	name2layers.insert(std::pair<std::string, std::vector<int>>("resnext101_32x8d", { 3, 4, 23, 3 }));

    return name2layers;
}

ResNet resnet18(int64_t num_classes);
ResNet resnet34(int64_t num_classes);
ResNet resnet50(int64_t num_classes);
ResNet resnet101(int64_t num_classes);

ResNet pretrained_resnet(int64_t num_classes, std::string model_name, std::string weight_path);
#endif // RESNET_H
