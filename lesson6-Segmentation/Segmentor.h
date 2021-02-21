#ifndef SEGMENTOR_H
#define SEGMENTOR_H
#include"UNet.h"
#include"SegDataset.h"
#include <io.h>
#include<opencv2/opencv.hpp>

template <class Model>
class Segmentor
{
public:
    Segmentor();
    void Initialize(int gpu_id, int width, int height,std::vector<std::string> name_list,
                    std::string encoder_name, std::string pretrained_path);
    void Train(float learning_rate, int epochs, int batch_size,
               std::string train_val_path, std::string image_type, std::string save_path);
    void LoadWeight(std::string weight_path);
    void Predict(cv::Mat image, std::string which_class);
private:
    int width = 512; int height = 512; std::vector<std::string> name_list;
    torch::Device device = torch::Device(torch::kCPU);
//    FPN fpn{nullptr};
//    UNet unet{nullptr};
    Model model{nullptr};
};

template <class Model>
Segmentor<Model>::Segmentor()
{
};

template <class Model>
void Segmentor<Model>::Initialize(int gpu_id,int _width, int _height, std::vector<std::string> _name_list,
                           std::string encoder_name, std::string pretrained_path){
    width = _width;
    height = _height;
    name_list = _name_list;
    if ((_access(pretrained_path.data(), 0)) == -1)
    {
        throw "Pretrained path is invalid";
    }
    if(name_list.size()<2) throw  "Class num is less than 1";
    int gpu_num = torch::getNumGPUs();
    if(gpu_id>=gpu_num) throw "GPU id exceeds max number of gpus";
    if(gpu_id>=0) device = torch::Device(torch::kCUDA, gpu_id);

    model = Model(name_list.size(),encoder_name,pretrained_path);
//    fpn = FPN(name_list.size(),encoder_name,pretrained_path);
    model->to(device);
}

template <class Model>
void Segmentor<Model>::Train(float learning_rate, int epochs, int batch_size,
                      std::string train_val_path, std::string image_type, std::string save_path){
    std::string train_dir = train_val_path+"\\train";
    std::string val_dir = train_val_path+"\\val";

    std::vector<std::string> list_images_train = {};
    std::vector<std::string> list_labels_train = {};
    std::vector<std::string> list_images_val = {};
    std::vector<std::string> list_labels_val = {};

    load_seg_data_from_folder(train_dir,image_type,list_images_train,list_labels_train);
    load_seg_data_from_folder(val_dir,image_type,list_images_val,list_labels_val);

    auto custom_dataset_train = SegDataset(width,height,list_images_train,list_labels_train,name_list).map(torch::data::transforms::Stack<>());
    auto custom_dataset_val = SegDataset(width,height,list_images_val,list_labels_val,name_list).map(torch::data::transforms::Stack<>());

    auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_train), batch_size);
    auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_val), batch_size);

    for(int epoch = 0; epoch<epochs; epoch++){
        float loss_sum = 0;
        int batch_count = 0;
        float loss_train = 0;
        float best_loss = 1e10;

        if(epoch == epochs/2) learning_rate/=10;
        torch::optim::Adam optimizer(model->parameters(), learning_rate);
        if(epoch <= epochs/2){
            for (auto mm : model->named_parameters())
            {
                if (strstr(mm.key().data(), "encoder"))
                {
                    mm.value().set_requires_grad(false);
                }
                else
                {
                    mm.value().set_requires_grad(true);
                }
            }
        }
        else{
            for (auto mm : model->named_parameters())
            {
                mm.value().set_requires_grad(true);
            }
        }
        model->train();
        for (auto& batch : *data_loader_train) {
            auto data = batch.data;
            auto target = batch.target;
            data = data.to(torch::kF32).to(device).div(255.0);
            target = target.to(torch::kLong).to(device).squeeze(1);//.clamp_max(1);//if you choose clamp, all classes will be set to only one

            optimizer.zero_grad();
            // Execute the model
            torch::Tensor prediction = model->forward(data);
            // Compute loss value
            torch::Tensor loss = torch::nll_loss2d(torch::log_softmax(prediction, /*dim=*/1), target);
            // Compute gradients
            loss.backward();
            // Update the parameters
            optimizer.step();
            loss_sum += loss.item().toFloat();
            batch_count++;
            loss_train = loss_sum / batch_count / batch_size;

            std::cout << "Epoch: " << epoch << "," << " Training Loss: " << loss_train << "\r";
        }
        std::cout<<std::endl;
        // validation part
        model->eval();
        loss_sum = 0; batch_count = 0;
        float loss_val = 0;
        for (auto& batch : *data_loader_val) {
            auto data = batch.data;
            auto target = batch.target;
            data = data.to(torch::kF32).to(device).div(255.0);
            target = target.to(torch::kLong).to(device).squeeze(1);//.clamp_max(1);

            // Execute the model
            torch::Tensor prediction = model->forward(data);

            // Compute loss value
            torch::Tensor loss = torch::nll_loss2d(torch::log_softmax(prediction, /*dim=*/1), target);
            loss_sum += loss.item<float>();
            batch_count++;
            loss_val = loss_sum / batch_count / batch_size;

            std::cout << "Epoch: " << epoch << "," << " Validation Loss: " << loss_val << "\r";
        }
        std::cout<<std::endl;
        if (loss_val < best_loss) {
            torch::save(model, save_path);
            best_loss = loss_val;
        }
    }
    return;
}

template <class Model>
void Segmentor<Model>::LoadWeight(std::string weight_path){
    torch::load(model, weight_path);
    model->eval();
    return;
}

template <class Model>
void Segmentor<Model>::Predict(cv::Mat image, std::string which_class){
    cv::Mat srcImg = image.clone();
    int which_class_index = -1;
    for(int i = 0; i<name_list.size(); i++){
        if(name_list[i] == which_class){
            which_class_index = i;
            break;
        }
    }
    if(which_class_index==-1) throw which_class + "not in the name list";
    int image_width = image.cols;
    int image_height = image.rows;
    cv::resize(image,image,cv::Size(width,height));
    torch::Tensor tensor_image = torch::from_blob(image.data, { 1, height, width,3 }, torch::kByte);
    tensor_image = tensor_image.to(device);
    tensor_image = tensor_image.permute({ 0,3,1,2 });
    tensor_image = tensor_image.to(torch::kFloat);
    tensor_image = tensor_image.div(255.0);

    at::Tensor output = model->forward({ tensor_image });
    output = torch::softmax(output, 1).mul(255.0).toType(torch::kByte);

    image = cv::Mat::ones(cv::Size(width, height), CV_8UC1);

    at::Tensor re = output[0][which_class_index].to(at::kCPU).detach();
    memcpy(image.data, re.data_ptr(), width * height * sizeof(unsigned char));
    cv::resize(image, image, cv::Size(image_width, image_height));

    // draw the prediction
	cv::imwrite("prediction.jpg", image);
    cv::imshow("prediction", image);
    cv::imshow("srcImage", srcImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}


#endif // SEGMENTOR_H
