#include <iostream>
#include <vector>
#include <fstream>
#include <map>

#include "mnist.h"

int main() {
    mnist::mnist_data images;
    if(!mnist::load_dataset(images, "images")) return 1;
    mnist::mnist_data labels;
    if(!mnist::load_dataset(labels, "labels")) return 1;
}