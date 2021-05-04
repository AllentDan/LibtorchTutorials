## Basic module construction
The idea of modular programming is very important. Modular programming can greatly reduce the repetitive coding process and increase the readability of the code. This chapter will describe how to use libtorch to build some basic modules of MLP and CNN.

### MLP basic unit
The first is the declaration and definition of the linear layer, including initialization and forward propagation functions. code show as below:
```cpp
class LinearBnReluImpl : public torch::nn::Module{
public:
    LinearBnReluImpl(int intput_features, int output_features);
    torch::Tensor forward(torch::Tensor x);
private:
    //layers
    torch::nn::Linear ln{nullptr};
    torch::nn::BatchNorm1d bn{nullptr};
};
TORCH_MODULE(LinearBnRelu);

LinearBnReluImpl::LinearBnReluImpl(int in_features, int out_features){
    ln = register_module("ln", torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features)));
    bn = register_module("bn", torch::nn::BatchNorm1d(out_features));
}

torch::Tensor LinearBnReluImpl::forward(torch::Tensor x){
    x = torch::relu(ln->forward(x));
    x = bn(x);
    return x;
}
```
When constructing the linear layer module class of MLP, we inherited the torch::nn::Module class, and made the initialization and forward propagation module as public The linear layer torch::nn::Linear and the normalization layer torch::nn::BatchNorm1d are hidden as private variables.

When defining the initialization function, you need to assign the original objects pointers ln and bn, and at the same time determine the names of the two. The forward propagation function is similar to forward in pytorch.

### CNN basic unit
The basic unit construction of CNN is similar to the construction of MLP, but it is slightly different. First, a defined time convolution hyperparameter determination function is required.
```cpp
inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 0, bool with_bias = false) {
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    conv_options.padding(padding);
    conv_options.bias(with_bias);
    return conv_options;
}
```
This function returns a torch::nn::Conv2dOptions object. The hyperparameters of the object are specified by the function, which is convenient to use. We use inline at the same time to improve code execution efficiency in Release mode.

Then it is similar to the linear module of MLP. The basic module of CNN consists of a convolutional layer, an activation function and a normalization layer. code show as below:
```cpp
class ConvReluBnImpl : public torch::nn::Module {
public:
    ConvReluBnImpl(int input_channel=3, int output_channel=64, int kernel_size = 3, int stride = 1);
    torch::Tensor forward(torch::Tensor x);
private:
    // Declare layers
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };
};
TORCH_MODULE(ConvReluBn);

ConvReluBnImpl::ConvReluBnImpl(int input_channel, int output_channel, int kernel_size, int stride) {
    conv = register_module("conv", torch::nn::Conv2d(conv_options(input_channel,output_channel,kernel_size,stride,kernel_size/2)));
    bn = register_module("bn", torch::nn::BatchNorm2d(output_channel));

}

torch::Tensor ConvReluBnImpl::forward(torch::Tensor x) {
    x = torch::relu(conv->forward(x));
    x = bn(x);
    return x;
}
```
## Simple MLP
In the MLP example, we take the construction of a four-layer perceptron as an example to introduce how to implement a deep learning model in cpp. The perceptron accepts features with in_features feature dims and outputs features with out_features feature dims. The number of intermediate features is defined as 32, 64 and 128. In fact, the reverse order is generally better, but it is not important as an example.
```cpp
class MLP: public torch::nn::Module{
public:
    MLP(int in_features, int out_features);
    torch::Tensor forward(torch::Tensor x);
private:
    int mid_features[3] = {32,64,128};
    LinearBnRelu ln1{nullptr};
    LinearBnRelu ln2{nullptr};
    LinearBnRelu ln3{nullptr};
    torch::nn::Linear out_ln{nullptr};
};

MLP::MLP(int in_features, int out_features){
    ln1 = LinearBnRelu(in_features, mid_features[0]);
    ln2 = LinearBnRelu(mid_features[0], mid_features[1]);
    ln3 = LinearBnRelu(mid_features[1], mid_features[2]);
    out_ln = torch::nn::Linear(mid_features[2], out_features);

    ln1 = register_module("ln1", ln1);
    ln2 = register_module("ln2", ln2);
    ln3 = register_module("ln3", ln3);
    out_ln = register_module("out_ln",out_ln);
}

torch::Tensor MLP::forward(torch::Tensor x){
    x = ln1->forward(x);
    x = ln2->forward(x);
    x = ln3->forward(x);
    x = out_ln->forward(x);
    return x;
}
```
The realization of each layer is through the basic module LinearBnRelu defined above.

## Simple CNN
The basic module ConvReluBn for building CNN was introduced earlier, and then I will try to build a CNN model with C++. The CNN is composed of three stages, and each stage is composed of a convolutional layer and a downsampling layer. This is equivalent to 8 times downsampling the original input image. The change in the number of channels in the middle layer is the same as the change in the feature number of the previous MLP, both of which are input->32->64->128->output.
```cpp
class plainCNN : public torch::nn::Module{
public:
    plainCNN(int in_channels, int out_channels);
    torch::Tensor forward(torch::Tensor x);
private:
    int mid_channels[3] = {32,64,128};
    ConvReluBn conv1{nullptr};
    ConvReluBn down1{nullptr};
    ConvReluBn conv2{nullptr};
    ConvReluBn down2{nullptr};
    ConvReluBn conv3{nullptr};
    ConvReluBn down3{nullptr};
    torch::nn::Conv2d out_conv{nullptr};
};

plainCNN::plainCNN(int in_channels, int out_channels){
    conv1 = ConvReluBn(in_channels,mid_channels[0],3);
    down1 = ConvReluBn(mid_channels[0],mid_channels[0],3,2);
    conv2 = ConvReluBn(mid_channels[0],mid_channels[1],3);
    down2 = ConvReluBn(mid_channels[1],mid_channels[1],3,2);
    conv3 = ConvReluBn(mid_channels[1],mid_channels[2],3);
    down3 = ConvReluBn(mid_channels[2],mid_channels[2],3,2);
    out_conv = torch::nn::Conv2d(conv_options(mid_channels[2],out_channels,3));

    conv1 = register_module("conv1",conv1);
    down1 = register_module("down1",down1);
    conv2 = register_module("conv2",conv2);
    down2 = register_module("down2",down2);
    conv3 = register_module("conv3",conv3);
    down3 = register_module("down3",down3);
    out_conv = register_module("out_conv",out_conv);
}

torch::Tensor plainCNN::forward(torch::Tensor x){
    x = conv1->forward(x);
    x = down1->forward(x);
    x = conv2->forward(x);
    x = down2->forward(x);
    x = conv3->forward(x);
    x = down3->forward(x);
    x = out_conv->forward(x);
    return x;
}
```
Assuming that a three-channel picture is input, the number of output channels is defined as n, and the input represents a tensor of [1,3,224,224], and an output tensor of [1,n,28,28] will be obtained.

## 简单LSTM
Simple LSTM
Finally, there is a simple LSTM example to deal with temporal features. Before directly using the torch::nn::LSTM class, let's first create a function that returns the torch::nn::LSTMOptions object. This function accepts the hyperparameters of the LSTM and returns the results defined by these hyperparameters.
```cpp
inline torch::nn::LSTMOptions lstmOption(int in_features, int hidden_layer_size, int num_layers, bool batch_first = false, bool bidirectional = false){
    torch::nn::LSTMOptions lstmOption = torch::nn::LSTMOptions(in_features, hidden_layer_size);
    lstmOption.num_layers(num_layers).batch_first(batch_first).bidirectional(bidirectional);
    return lstmOption;
}

//batch_first: true for io(batch, seq, feature) else io(seq, batch, feature)
class LSTM: public torch::nn::Module{
public:
    LSTM(int in_features, int hidden_layer_size, int out_size, int num_layers, bool batch_first);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear ln{nullptr};
    std::tuple<torch::Tensor, torch::Tensor> hidden_cell;
};
```
After declaring the LSTM, we will implement the internal initialization function and forward propagation function as follows:
```cpp
LSTM::LSTM(int in_features, int hidden_layer_size, int out_size, int num_layers, bool batch_first){
    lstm = torch::nn::LSTM(lstmOption(in_features, hidden_layer_size, num_layers, batch_first));
    ln = torch::nn::Linear(hidden_layer_size, out_size);

    lstm = register_module("lstm",lstm);
    ln = register_module("ln",ln);
}

torch::Tensor LSTM::forward(torch::Tensor x){
    auto lstm_out = lstm->forward(x);
    auto predictions = ln->forward(std::get<0>(lstm_out));
    return predictions.select(1,-1);
}
```