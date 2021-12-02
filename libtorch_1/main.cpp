#include <iostream>
#include<torch/script.h>
#include<torch/torch.h>
#include <vector>
#include <torch/types.h>

using namespace std;

void createTensor() {
    //第一种，固定尺寸和值的初始化，通过 torch::kCUDA转移到gpu
    torch::Tensor a = torch::ones({3, 4}, torch::kCUDA);
    std::cout << "a = torch::ones({3,4})" << endl << a << std::endl;
    auto options = torch::TensorOptions().device(torch::kCUDA);
    auto b = torch::eye(3, options);
    std::cout << "torch::eye(3)" << endl << b << std::endl;
    auto c = torch::tensor({1, 2, 3, 4}, torch::kCUDA);
    cout << "torch::tensor({1,2,3,4})" << endl << c << endl;
    //从c++的其他数据类型转换而来
    std::vector<int> vec_a = {1, 2, 3, 4, 5, 6};
    torch::Tensor t_a = torch::from_blob(vec_a.data(), {2, 3}, torch::kI32);
    cout << "t_a" << endl << t_a << endl;
    float vec_b[9]{9.0, 8, 7, 6, 5, 3, 2, 1.0, 0};
    torch::Tensor t_b = torch::from_blob(vec_b, {3, 3}, torch::kFloat32);
    cout << "t_b" << endl << t_b << endl;
}

void reshapeTensor() {
    auto a = torch::full({10}, 3);
    a = a.to(torch::kCUDA);
    cout << "a:" << endl << a << endl;
    a = a.view({1, 2, -1});
    cout << "after a : " << endl << a << endl;
}

void selectTensor() {
    torch::Tensor tensor_a = torch::rand({10, 3, 28, 28});
    cout << tensor_a[0].sizes() << endl;
    cout << tensor_a[0][0].sizes() << endl;
    //选择第0维的0，3，3组成新张量[3,3,28,28]
    std::cout << tensor_a.index_select(0, torch::tensor({0, 3, 3})).sizes() << endl;
    //选择第1维的第0和第2的组成新张量[10, 2, 28, 28]
    std::cout << tensor_a.index_select(1, torch::tensor({0, 2})).sizes() << endl;
    //选择第1维，从0开始，截取长度为2的部分张量[10, 2, 28, 28]
    std::cout << tensor_a.narrow(1, 0, 2).sizes() << endl;
    //选择第3维度的第二个张量，即所有图片的第2行组成的张量[10, 3, 28]
    std::cout << tensor_a.select(3, 2).sizes();

    //use for index
    torch::Tensor ten_index = torch::randn({3, 4});
    torch::Tensor mask = torch::zeros({3, 4});
    mask[0][1] = 1;
    std::cout << "ten_index" << endl << ten_index << endl;
    std::cout << "index: " << endl << ten_index.index({mask.to(torch::kBool)}) << endl;
}

void catTensor() {
    torch::Tensor cat_a = torch::ones({3, 4});
    torch::Tensor cat_b = torch::ones({3, 4});
    torch::Tensor cat_c = torch::cat({cat_a, cat_b}, 1);
    cout << cat_c << endl;
}

int main() {
    std::cout << "Hello, World!" << std::endl;
//    createTensor();
//    reshapeTensor();
//    selectTensor();
    catTensor();
    return 0;
}
