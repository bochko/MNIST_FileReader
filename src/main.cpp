#include <iostream>
#include <vector>
#include <fstream>

#define MNIST_IMAGE_ROW_COUNT   28
#define MNIST_IMAGE_COL_COUNT   28

typedef struct {
    std::uint32_t magic;
    std::uint32_t sample_count;
    std::vector<std::uint32_t> dimension_sizes;
    std::vector<std::uint8_t*> data;
} MnistData;

bool arch_is_lsb(void)
{
    volatile std::uint32_t val = 0x01234567;
    return(*((std::uint8_t*)(&val))) == 0x67;
}

std::uint32_t revb_int32(std::uint32_t val)
{
    val =   (val >> 24u) | ((val << 8u) & 0x00FF0000)
            | ((val >> 8u) & 0x0000FF00) | (val << 24u);
    return val;
}



void load_dataset(MnistData& data, std::string which)
{
    std::cout << "Enter " << which << " dataset filepath:" << std::endl;
    std::string training_set_path;
    std::getline(std::cin, training_set_path);

    std::ifstream file;
    file.open(training_set_path);
    // read magic
    file.read((char*) &data.magic, sizeof(std::uint32_t));
    // read sample count
    file.read((char*) &data.sample_count, sizeof(std::uint32_t));
    // interpret magic number
    if(arch_is_lsb())
    {
        data.magic = revb_int32(data.magic);
        data.sample_count = revb_int32(data.sample_count);
    }
    // extract dimensions
    uint32_t dimension_mask = 0x000000FF;
    uint32_t dimension_count = (data.magic & dimension_mask) - 1U; // sample_size is always first dimension

    for(int i = 0; i < dimension_count; i++)
    {
        uint32_t dimension_size;
        file.read((char*) &dimension_size, sizeof(std::uint32_t));
        if(arch_is_lsb())
        {
            dimension_size = revb_int32(dimension_size);
        }
        data.dimension_sizes.push_back(dimension_size);
    }

    // calculate data memory footprint per sample
    uint32_t memory_footprint = 1;
    for(int i = 0; i < data.dimension_sizes.size(); i++)
    {
        memory_footprint *= data.dimension_sizes.at(i);
    }
    // extract data
    for (int i = 0; i < data.sample_count; i++)
    {

        uint8_t* sample = new uint8_t[memory_footprint];
        file.read((char*) sample, memory_footprint);
        data.data.push_back(sample);
    }
}

int main() {
    MnistData images;
    load_dataset(images, "images");
    MnistData labels;
    load_dataset(labels, "labels");
}