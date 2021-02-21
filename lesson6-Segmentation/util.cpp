#include "util.h"

SegmentationHeadImpl::SegmentationHeadImpl(int in_channels, int out_channels, int kernel_size, double _upsampling){
    conv2d = torch::nn::Conv2d(conv_options(in_channels, out_channels, kernel_size, 1, kernel_size / 2));
    upsampling = torch::nn::Upsample(upsample_options(std::vector<double>{_upsampling,_upsampling}));
    register_module("conv2d",conv2d);
}
torch::Tensor SegmentationHeadImpl::forward(torch::Tensor x){
    x = conv2d->forward(x);
    x = upsampling->forward(x);
    return x;
}

std::string replace_all_distinct(std::string str, const std::string old_value, const std::string new_value)
{
    for (std::string::size_type pos(0); pos != std::string::npos; pos += new_value.length())
    {
        if ((pos = str.find(old_value, pos)) != std::string::npos)
        {
            str.replace(pos, old_value.length(), new_value);
        }
        else { break; }
    }
    return   str;
}

//遍历该目录下的.xml文件，并且找到对应的
void load_seg_data_from_folder(std::string folder, std::string image_type,
                               std::vector<std::string> &list_images, std::vector<std::string> &list_labels)
{
    long long hFile = 0; //句柄
    struct _finddata_t fileInfo;
    std::string pathName;
    if ((hFile = _findfirst(pathName.assign(folder).append("\\*.*").c_str(), &fileInfo)) == -1)
    {
        return;
    }
    do
    {
        const char* s = fileInfo.name;

        if (fileInfo.attrib&_A_SUBDIR) //是子文件夹
        {
            //遍历子文件夹中的文件(夹)，查看是否有txt文件
            if (strcmp(s, ".") == 0 || strcmp(s, "..") == 0) //子文件夹目录是.或者..
                continue;
            std::string sub_path = folder + "\\" + fileInfo.name;
            load_seg_data_from_folder(sub_path, image_type, list_images, list_labels);
        }
        else //判断是不是txt文件
        {

            if (strstr(s, ".json"))
            {
                std::string label_path = folder + "\\" + fileInfo.name;
                list_labels.push_back(label_path);
                std::string image_path = replace_all_distinct(label_path, ".json", image_type);
                list_images.push_back(image_path);
            }
        }
    } while (_findnext(hFile, &fileInfo) == 0);
    return;
}
