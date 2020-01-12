#include <iostream>
#include <vector>
#include <fstream>
#include <map>

namespace mnist {
#define MNIST_IMAGE_ROW_COUNT   28
#define MNIST_IMAGE_COL_COUNT   28

    typedef enum {
        MNIST_DATATYPE_NONE = 0x00,
        MNIST_UNSIGNED_BYTE = 0x08,
        MNIST_SIGNED_BYTE = 0x09,
        MNIST_SHORT = 0x0B,
        MNIST_INT = 0x0C,
        MNIST_FLOAT = 0x0D,
        MNIST_DOUBLE = 0x0E
    } MnistDatatype;

    typedef struct {
        std::uint32_t magic;
        MnistDatatype datatype;
        std::uint8_t width;
        std::uint32_t sample_count;
        std::vector<std::uint32_t> dimension_sizes;
        std::vector<std::uint8_t *> data;
    } MnistData;

    const static std::map<MnistDatatype, uint8_t> trivial_sizes = {
            {MNIST_UNSIGNED_BYTE, 1},
            {MNIST_SIGNED_BYTE,   1},
            {MNIST_SHORT,         2},
            {MNIST_INT,           4},
            {MNIST_FLOAT,         4},
            {MNIST_DOUBLE,        8}
    };

    bool arch_is_lsb(void) {
        volatile std::uint32_t val = 0x01234567;
        return (*((std::uint8_t *) (&val))) == 0x67;
    }

    std::uint32_t revb_int32(std::uint32_t val) {
        val = (val >> 24u) | ((val << 8u) & 0x00FF0000)
              | ((val >> 8u) & 0x0000FF00) | (val << 24u);
        return val;
    }

    bool open_ifstream(std::ifstream &file, std::string filepath) {
        bool open = true;
        try {
            file.open(filepath);
            if (!file.is_open()) {
                open = false;
            }
        } catch (std::exception &e) {
            std::cerr << e.what();
            open = false;
        }
        return open;
    }

    void load_dataset(MnistData &data, std::string which) {
        std::cout << "Enter " << which << " dataset filepath:" << std::endl;
        std::string training_set_path;
        std::getline(std::cin, training_set_path);
        std::ifstream file;
        if (!open_ifstream(file, training_set_path)) {
            return;
        }
        file.read((char *) &data.magic, sizeof(std::uint32_t)); // read magic
        file.read((char *) &data.sample_count, sizeof(std::uint32_t)); // read sample count
        if (arch_is_lsb()) // interpret magic number
        {
            data.magic = revb_int32(data.magic);
            data.sample_count = revb_int32(data.sample_count);
        }
        uint32_t dimension_mask = 0x000000FF; // extract dimensions
        uint32_t dimension_count = (data.magic & dimension_mask) - 1U; // sample_size is always first dimension
        for (uint32_t i = 0; i < dimension_count; i++) {
            uint32_t dimension_size;
            file.read((char *) &dimension_size, sizeof(std::uint32_t));
            if (arch_is_lsb()) {
                dimension_size = revb_int32(dimension_size);
            }
            data.dimension_sizes.push_back(dimension_size);
        }
        uint32_t datatype_mask = 0x0000000F; // extract datatype
        uint32_t datatype = (data.magic >> 8u) & datatype_mask;
        if(trivial_sizes.find((MnistDatatype)datatype) != trivial_sizes.end())
        {
            data.datatype = (MnistDatatype)datatype;
            data.width = trivial_sizes.at((MnistDatatype)datatype);
        }
        else
        {
            data.datatype = MNIST_DATATYPE_NONE;
            data.width = 0u;
        }
        // calculate data memory footprint per sample
        uint32_t memory_footprint = 1;
        for (std::uint32_t dimension_size : data.dimension_sizes) {
            memory_footprint *= dimension_size;
        }
        memory_footprint *= data.width;
        // extract data
        for (uint32_t i = 0; i < data.sample_count; i++) {
            auto *sample = new uint8_t[memory_footprint];
            file.read((char *) sample, memory_footprint);
            data.data.push_back(sample);
        }
    }


}

int main() {
    mnist::MnistData images;
    mnist::load_dataset(images, "images");
    mnist::MnistData labels;
    mnist::load_dataset(labels, "labels");
}