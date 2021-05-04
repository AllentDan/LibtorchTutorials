This chapter will introduce in detail how to use libtorch's own data loading module, which is an important condition for model training. Unless the function of this data loading module is not enough, it is still necessary to inherit the data loading class of libtorch, which is simple and efficient.

## Preconditions
Libtorch provides a wealth of base classes for users to customize derived classes. Torch::data::Dataset is one of the commonly used base classes. To use this class, you need to understand the base class and derived classes, as well as the so-called inheritance and polymorphism. Anyone with C++ programming experience should be familiar with it. For the convenience of readers at different stages, I will briefly explain it. Class is the father, which can give birth to different sons. The birth of sons is called derivation or inheritance (depending on the context of use), and the birth of different sons achieves polymorphism. The father is the base class, and the son is the derived class. In reality, some fathers will leave part of his own property for the aged, and the sons can’t touch it. This is private. Part of the property can be used by the son, but the son’s instance(wife) cannot be used. This is called protected, and some property can be used by anyone then the public. Similar to the parent and child in reality, in the code, the derived class can use part of the properties or functions of the parent class, depending on how the parent class is defined.

Then we have to understand the virtual function, that is, the father specifies that part of the property is public, but it can only be used to buy a house. Different sons can buy different houses through different ways. This means that the property is virtual in the father. Subclasses who want to inherit this virtual property can re-plan their usage.

In fact, if you have programming experience with pytorch, you will soon find that the use of libtorch's Dataset class is very similar to the use of python. Pytorch custom dataload needs to define a derived class of Dataset, including initialization function __init__, get function __getitem__ and data set size function __len__. Similarly, the initialization function, get() function and size() function also need to be used in libtorch.

## Picture file traversal
We will use classification task as an example to introduce the use of libtorch's Dataset class. Click the [insect classification data set](https://download.pytorch.org/tutorial/hymenoptera_data.zip) provided by the pytorch official website, download it to the local and decompress it. The root directory of the data set is used as an index to realize the loading of images by Dataloader.

First we define a function to load pictures, use the C++ slightly modify the code as follows. It is not perfect and the code quality can be improved but enough as an example.
```cpp
//traverse all files ends with type
void load_data_from_folder(std::string image_dir, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label);

void load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label)
{
    long long hFile = 0;
    struct _finddata_t fileInfo;
    std::string pathName;
    if ((hFile = _findfirst(pathName.assign(path).append("\\*.*").c_str(), &fileInfo)) == -1)
    {
        return;
    }
    do
    {
        const char* s = fileInfo.name;
        const char* t = type.data();

        if (fileInfo.attrib&_A_SUBDIR) //is sub folder
        {
            //traverse sub folder
            if (strcmp(s, ".") == 0 || strcmp(s, "..") == 0) //if sub folder is . or ..
                continue;
            std::string sub_path = path + "\\" + fileInfo.name;
            label++;
            load_data_from_folder(sub_path, type, list_images, list_labels, label);

        }
        else //check if the fild ends with type
        {
            if (strstr(s, t))
            {
                std::string image_path = path + "\\" + fileInfo.name;
                list_images.push_back(image_path);
                list_labels.push_back(label);
            }
        }
    } while (_findnext(hFile, &fileInfo) == 0);
    return;
}
```
The modified function accepts the data set folder path image_dir and image type image_type, stores the traversed image path and its category in list_images and list_labels respectively. And finally the label variable is used to represent the category count. Pass in label=-1, and the returned label value plus one is equal to the picture category.

## Custom dataset
Define dataSetClc, which is inherited from torch::data::Dataset. Define private variables image_paths and labels to store the image path and category respectively, which are two vector variables. The initialization function of dataSetClc is to load pictures and categories. Return a tensor list consisting of images and categories through the get() function. You can do any image-oriented operations in the get() function, such as data enhancement. The effect is equivalent to the data enhancement in \_\_getitem__ in pytorch.
```cpp
class dataSetClc:public torch::data::Dataset<dataSetClc>{
public:
    int class_index = 0;
    dataSetClc(std::string image_dir, std::string type){
        load_data_from_folder(image_dir, std::string(type), image_paths, labels, class_index-1);
    }
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override{
        std::string image_path = image_paths.at(index);
        cv::Mat image = cv::imread(image_path);
        cv::resize(image, image, cv::Size(224, 224)); //uniform the image size for tensor stacking
        int label = labels.at(index);
        torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
        torch::Tensor label_tensor = torch::full({ 1 }, label);
        return {img_tensor.clone(), label_tensor.clone()};
    }
    // Override size() function, return the length of data
    torch::optional<size_t> size() const override {
        return image_paths.size();
    };
private:
    std::vector<std::string> image_paths;
    std::vector<int> labels;
};
```

## Use a custom Dataset
In the following, we will show how to use the defined data loading class, and the training set in the insect classification as a test. The code is as follows. You can print the loaded image tensor and category.
```cpp
int batch_size = 2;
std::string image_dir = "your path to\\hymenoptera_data\\train";
auto mdataset = myDataset(image_dir,".jpg").map(torch::data::transforms::Stack<>());
auto mdataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(mdataset), batch_size);
for(auto &batch: *mdataloader){
    auto data = batch.data;
    auto target = batch.target;
    std::cout<<data.sizes()<<target;
}
```