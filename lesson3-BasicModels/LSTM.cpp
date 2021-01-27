#include<LSTM.h>

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
