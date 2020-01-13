#include <iostream>
#include <vector>
#include <fstream>
#include <map>

#include "mnist.h"

int main() {
    mnist::mnist_data images;
    // print prompt
    std::cout << "Enter images dataset filepath:" << std::endl;
    std::string filepath;
    std::getline(std::cin, filepath);
    if(!mnist::load_dataset(images, filepath)) return 1;

    mnist::mnist_data labels;
    // print prompt
    std::cout << "Enter images dataset filepath:" << std::endl;
    std::getline(std::cin, filepath);
    if(!mnist::load_dataset(labels, filepath)) return 1;
}