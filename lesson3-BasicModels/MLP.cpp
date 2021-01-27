#include<MLP.h>

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
