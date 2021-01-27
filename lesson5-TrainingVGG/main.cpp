#include "mainwindow.h"

#include <QApplication>
#include<Classification.h>

int main(int argc, char *argv[])
{
    auto pavgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));
    auto inp = torch::rand({1,3,7,7});
    auto outp = pavgpool->forward(inp);
    std::cout<<outp.sizes();
    std::vector<int> cfg_dd = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    auto vgg_dd = VGG(cfg_dd,1000,true);
    auto in = torch::rand({1,3,224,224});
    auto dictdd = vgg_dd->named_parameters();
    vgg_dd->forward(in);
    for (auto n = dictdd.begin(); n != dictdd.end(); n++)
    {
        std::cout<<(*n).key()<<std::endl;
    }

    std::string vgg_path = "D:\\AllentFiles\\code\\tmp\\vgg16_bn.pt";
    std::string train_val_dir = "D:\\AllentFiles\\data\\dataset4teach\\hymenoptera_data";
    Classifier classifier(0);
    classifier.Initialize(2,vgg_path);

    //predict
    classifier.LoadWeight("classifer.pt");
    cv::Mat image = cv::imread(train_val_dir+"\\val\\bees\\2407809945_fb525ef54d.jpg");
    classifier.Predict(image);
    classifier.Train(300,4,0.0003,train_val_dir,".jpg","classifer.pt");
    std::vector<int> cfg_a = {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1};
    std::vector<int> cfg_d = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    auto vgg = VGG(cfg_d,1000,true);
    auto dict = vgg->named_parameters();
    torch::load(vgg, vgg_path);
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
