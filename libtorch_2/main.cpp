#include <iostream>
#include <torch/torch.h>
#include <torch/nn/module.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "include/CNN.h"
#include <string>

using namespace std;

int main() {
//    std::cout << "Hello, World!" << std::endl;
    torch::DeviceType device_type;

    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    auto cnn = plainCNN(3, 1);
    cnn.to(device);
    auto cnn_input = torch::randint(255, {1, 3, 224, 224}).to(device);
    torch::optim::Adam optimizer_cnn(cnn.parameters(), 0.0003);
    auto cnn_target = torch::zeros({1, 1, 26, 26}).to(device);
    for (int i = 0; i < 30; i++) {
        optimizer_cnn.zero_grad();
        auto out = cnn.forward(cnn_input);
        auto loss = torch::mse_loss(out, cnn_target);
        loss.backward();
        optimizer_cnn.step();
        cout << out[0][0][0];
    }
    
    return 0;
}
