#include "Classification.h"


Classifier::Classifier(int gpu_id)
{
    if (gpu_id >= 0) {
        device = torch::Device(torch::kCUDA, gpu_id);
    }
    else {
        device = torch::Device(torch::kCPU);
    }
}

void Classifier::Initialize(int _num_classes, std::string _pretrained_path){
    std::vector<int> cfg_d = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    auto net_pretrained = VGG(cfg_d,1000,true);
    vgg = VGG(cfg_d,_num_classes,true);
    torch::load(net_pretrained, _pretrained_path);
    torch::OrderedDict<std::string, at::Tensor> pretrained_dict = net_pretrained->named_parameters();
    torch::OrderedDict<std::string, at::Tensor> model_dict = vgg->named_parameters();

    for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++)
    {
        if (strstr((*n).key().data(), "classifier")) {
            continue;
        }
        model_dict[(*n).key()] = (*n).value();
    }

    torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
    auto new_params = model_dict; // implement this
    auto params = vgg->named_parameters(true /*recurse*/);
    auto buffers = vgg->named_buffers(true /*recurse*/);
    for (auto& val : new_params) {
        auto name = val.key();
        auto* t = params.find(name);
        if (t != nullptr) {
            t->copy_(val.value());
        }
        else {
            t = buffers.find(name);
            if (t != nullptr) {
                t->copy_(val.value());
            }
        }
    }
    torch::autograd::GradMode::set_enabled(true);
    try
    {
        vgg->to(device);
    }
    catch (const std::exception&e)
    {
        std::cout << e.what() << std::endl;
    }

    return;
}

void Classifier::Train(int num_epochs, int batch_size, float learning_rate, std::string train_val_dir, std::string image_type, std::string save_path){
    std::string path_train = train_val_dir+ "\\train";
    std::string path_val = train_val_dir + "\\val";

    auto custom_dataset_train = dataSetClc(path_train, image_type).map(torch::data::transforms::Stack<>());
    auto custom_dataset_val = dataSetClc(path_val, image_type).map(torch::data::transforms::Stack<>());

    auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_train), batch_size);
    auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_val), batch_size);

    float loss_train = 0; float loss_val = 0;
    float acc_train = 0.0; float acc_val = 0.0; float best_acc = 0.0;
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        size_t batch_index_train = 0;
        size_t batch_index_val = 0;
        if (epoch == int(num_epochs / 2)) { learning_rate /= 10; }
        torch::optim::Adam optimizer(vgg->parameters(), learning_rate); // Learning Rate
        if (epoch < int(num_epochs / 8))
        {
            for (auto mm : vgg->named_parameters())
            {
                if (strstr(mm.key().data(), "classifier"))
                {
                    mm.value().set_requires_grad(true);
                }
                else
                {
                    mm.value().set_requires_grad(false);
                }
            }
        }
        else {
            for (auto mm : vgg->named_parameters())
            {
                mm.value().set_requires_grad(true);
            }
        }
        // Iterate data loader to yield batches from the dataset
        for (auto& batch : *data_loader_train) {
            auto data = batch.data;
            auto target = batch.target.squeeze();
            data = data.to(torch::kF32).to(device).div(255.0);
            target = target.to(torch::kInt64).to(device);
            optimizer.zero_grad();
            // Execute the model
            torch::Tensor prediction = vgg->forward(data);
            //cout << prediction << endl;
            auto acc = prediction.argmax(1).eq(target).sum();
            acc_train += acc.template item<float>() / batch_size;
            // Compute loss value
            torch::Tensor loss = torch::nll_loss(prediction, target);
            // Compute gradients
            loss.backward();
            // Update the parameters
            optimizer.step();
            loss_train += loss.item<float>();
            batch_index_train++;
            std::cout << "Epoch: " << epoch << " |Train Loss: " << loss_train / batch_index_train << " |Train Acc:" << acc_train / batch_index_train << "\r";
        }
        std::cout << std::endl;

        //validation part
        vgg->eval();
        for (auto& batch : *data_loader_val) {
            auto data = batch.data;
            auto target = batch.target.squeeze();
            data = data.to(torch::kF32).to(device).div(255.0);
            target = target.to(torch::kInt64).to(device);
            torch::Tensor prediction = vgg->forward(data);
            // Compute loss value
            torch::Tensor loss = torch::nll_loss(prediction, target);
            auto acc = prediction.argmax(1).eq(target).sum();
            acc_val += acc.template item<float>() / batch_size;
            loss_val += loss.item<float>();
            batch_index_val++;
            std::cout << "Epoch: " << epoch << " |Val Loss: " << loss_val / batch_index_val << " |Valid Acc:" << acc_val / batch_index_val << "\r";
        }
        std::cout << std::endl;


        if (acc_val > best_acc) {
            torch::save(vgg, save_path);
            best_acc = acc_val;
        }
        loss_train = 0; loss_val = 0; acc_train = 0; acc_val = 0; batch_index_train = 0; batch_index_val = 0;
    }
}

int Classifier::Predict(cv::Mat& image){
    cv::resize(image, image, cv::Size(448, 448));
    torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 });
    img_tensor = img_tensor.to(device).unsqueeze(0).to(torch::kF32).div(255.0);
    auto prediction = vgg->forward(img_tensor);
    prediction = torch::softmax(prediction,1);
    auto class_id = prediction.argmax(1);
    std::cout<<prediction<<class_id;
    int ans = int(class_id.item().toInt());
    float prob = prediction[0][ans].item().toFloat();
    return ans;
}

void Classifier::LoadWeight(std::string weight){
    torch::load(vgg,weight);
    vgg->eval();
    return;
}
