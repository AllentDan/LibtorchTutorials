#ifndef MLP_H
#define MLP_H
#endif // MLP_H
#include<BaseModule.h>

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
