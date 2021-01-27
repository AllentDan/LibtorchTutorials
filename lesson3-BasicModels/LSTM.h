#ifndef LSTM_H
#define LSTM_H

#endif // LSTM_H
#include<BaseModule.h>
#include<tuple>

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
