#include "mainwindow.h"
#include<opencv2/opencv.hpp>
#include<CNN.h>
#include<MLP.h>
#include<LSTM.h>
#include <QApplication>
#include<iostream>
#include<string>
using namespace std;

int main(int argc, char *argv[])
{
    //LSTM part
    auto lstm = LSTM(3, 100, 2,1,true);
    auto lstm_input = torch::Tensor(torch::linspace(1,15,15)).unsqueeze(1).repeat({1,3}).unsqueeze(0).repeat({4,1,1});//[4,15,3]
    auto lstm_target = torch::full({4,2},16).to(torch::kFloat);
    auto optimizer_lstm = torch::optim::Adam(lstm.parameters(),0.003);
    for(int i=0; i<130;i++){
        optimizer_lstm.zero_grad();
        auto out = lstm.forward(lstm_input.to(torch::kFloat));
        auto loss = torch::mse_loss(out,lstm_target);
        loss.backward();
        optimizer_lstm.step();
        cout<<out;
    }


    //CNN part
    auto cnn = plainCNN(3,1);
    auto cnn_input = torch::randint(255,{1,3,224,224});
    torch::optim::Adam optimizer_cnn(cnn.parameters(), 0.0003);
    auto cnn_target = torch::zeros({1,1,26,26});
    for(int i=0; i<30;i++){
        optimizer_cnn.zero_grad();
        auto out = cnn.forward(cnn_input);
        auto loss = torch::mse_loss(out,cnn_target);
        loss.backward();
        optimizer_cnn.step();
        cout<<out[0][0][0];
    }

    auto mlp = MLP(10,1);
    auto mlp_input = torch::rand({2,10});
    auto mlp_target = torch::ones({2,1});
    torch::optim::Adam optimizer_mlp(mlp.parameters(), 0.0005);
    for(int i=0; i<400; i++){
        optimizer_mlp.zero_grad();
        auto out = mlp.forward(mlp_input);
        auto loss = torch::mse_loss(out,mlp_target);
        loss.backward();
        optimizer_mlp.step();
        cout<<out;
    }

    string pt_path = "D:/AllentFiles/code/tmp/resnet34.pt";
    torch::jit::Module jit_model = torch::jit::load(pt_path);
    //auto input = torch::zeros({1,3,224,224},torch::kFloat);
    //auto output = jit_model.forward({input});

    auto model = ConvReluBn(3,4,3);
    auto input = torch::zeros({1,3,12,12},torch::kFloat);
    auto input2 = torch::zeros({1,3,12,12},torch::kFloat);
    auto output = input*input2;
    output = model->forward(input);
    std::cout<<output.sizes()<<std::endl;
    std::cout.flush();

    std::cout<<"Test output";
    cv::Mat image = cv::imread("C:\\Users\\Administrator\\Pictures\\1.jpg");
    cv::Mat M(200, 200, CV_8UC3, cv::Scalar(0, 0, 255));
    if(!M.data)
        return 0;
    cv::imshow("fff",image);
    cv::imshow("ddd",M);
    cv::waitKey(0);
    cv::destroyAllWindows();
    std::cout<<image.size();

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
