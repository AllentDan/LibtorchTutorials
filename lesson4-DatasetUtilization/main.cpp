#include "mainwindow.h"

#include <QApplication>
#include<dataSet.h>

int main(int argc, char *argv[])
{
    int batch_size = 2;
    std::string image_dir = "D:\\AllentFiles\\data\\dataset4teach\\hymenoptera_data\\train";
    auto mdataset = myDataset(image_dir,".jpg").map(torch::data::transforms::Stack<>());
    auto mdataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(mdataset), batch_size);
    for(auto &batch: *mdataloader){
        auto data = batch.data;
        auto target = batch.target;
        std::cout<<data.sizes()<<target;
    }

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
