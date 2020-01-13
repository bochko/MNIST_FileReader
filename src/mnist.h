#ifndef MNIST_MACHINELEARNING_MNIST_H
#define MNIST_MACHINELEARNING_MNIST_H

#include <cstdint>
#include <vector>

namespace mnist {
    typedef enum {
        MNIST_DATATYPE_NONE = 0x00,
        MNIST_UNSIGNED_BYTE = 0x08,
        MNIST_SIGNED_BYTE = 0x09,
        MNIST_SHORT = 0x0B,
        MNIST_INT = 0x0C,
        MNIST_FLOAT = 0x0D,
        MNIST_DOUBLE = 0x0E
    } mnist_datatype;

    typedef struct {
        std::uint32_t magic;
        mnist_datatype datatype;
        std::uint8_t width;
        std::uint32_t sample_count;
        std::vector<std::uint32_t> dimension_sizes;
        std::vector<std::uint8_t *> data;
    } mnist_data;

    bool load_dataset(mnist_data &data, const std::string& filepath);

    namespace util {
        bool architecture_is_lsb();
        std::uint32_t revb_uint32(std::uint32_t val);
    }
}

#endif //MNIST_MACHINELEARNING_MNIST_H
