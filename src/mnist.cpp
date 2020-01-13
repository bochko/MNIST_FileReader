//
// Created by Boyan Atanasov on 12/01/2020.
//
#include <map>
#include <iostream>
#include <fstream>

#include "mnist.h"

static bool open_ifstream(std::ifstream &file, std::string filepath) noexcept;

static bool read_ifstream_safe(std::ifstream &file, char *destination, std::size_t size) noexcept;

const static std::map<mnist::mnist_datatype, uint8_t> trivial_sizes = {
        {mnist::MNIST_UNSIGNED_BYTE, 1},
        {mnist::MNIST_SIGNED_BYTE,   1},
        {mnist::MNIST_SHORT,         2},
        {mnist::MNIST_INT,           4},
        {mnist::MNIST_FLOAT,         4},
        {mnist::MNIST_DOUBLE,        8}
};

bool mnist::util::architecture_is_lsb()
{
    volatile std::uint32_t val = 0x01234567;
    return (*(reinterpret_cast<volatile std::uint8_t *>(&val))) == 0x67;
}

std::uint32_t mnist::util::revb_uint32(std::uint32_t val)
{
    val = (val >> 24u) | ((val << 8u) & 0x00FF0000)
          | ((val >> 8u) & 0x0000FF00) | (val << 24u);
    return val;
}

static bool open_ifstream(std::ifstream &file, std::string filepath) noexcept
{
    bool open = true;
    try
    {
        file.open(filepath);
        if (!file.is_open())
        {
            open = false;
        }
    } catch (std::exception &e)
    {
        std::cerr << e.what();
        open = false;
    }
    return open;
}

static bool read_ifstream_safe(std::ifstream &file, char *destination, std::size_t size) noexcept
{
    bool read = true;
    try
    {
        file.read(destination, size);
    } catch (std::exception &e)
    {
        std::cerr << e.what();
        read = false;
    }
    return read;
}

bool mnist::load_dataset(mnist_data &data, std::string which)
{
    // print prompt
    std::cout << "Enter " << which << " dataset filepath:" << std::endl;
    std::string training_set_path;
    std::getline(std::cin, training_set_path);
    // open file @ filepath if possible, if not, return
    std::ifstream file;
    if (!open_ifstream(file, training_set_path))
    {
        std::cerr << "Could not open file at \"" << training_set_path << "\"" << std::endl;
        return false;
    }
    // read magic
    if (!(read_ifstream_safe(file, reinterpret_cast<char *>(&data.magic),
                             sizeof(std::uint32_t))))
        return false;
    // read sample count
    if (!(read_ifstream_safe(file, reinterpret_cast<char *>(&data.sample_count),
                             sizeof(std::uint32_t))))
        return false;
    // interpret magic number
    if (util::architecture_is_lsb())
    {
        data.magic = util::revb_uint32(data.magic);
        data.sample_count = util::revb_uint32(data.sample_count);
    }
    uint32_t dimension_mask = 0x000000FF; // extract dimensions
    uint32_t dimension_count =
            (data.magic & dimension_mask) - 1U; // sample_size is always first dimension
    for (uint32_t i = 0; i < dimension_count; i++)
    {
        uint32_t dimension_size;
        if (!(read_ifstream_safe(file, reinterpret_cast<char *>(&dimension_size), sizeof(std::uint32_t))))
            return false;
        if (util::architecture_is_lsb())
        {
            dimension_size = util::revb_uint32(dimension_size);
        }
        data.dimension_sizes.push_back(dimension_size);
    }
    uint32_t datatype_mask = 0x0000000F; // extract datatype
    uint32_t datatype = (data.magic >> 8u) & datatype_mask;
    if (trivial_sizes.find(static_cast<mnist_datatype>(datatype)) != trivial_sizes.end())
    {
        data.datatype = static_cast<mnist_datatype>(datatype);
        data.width = trivial_sizes.at(static_cast<mnist_datatype>(datatype));
    } else
    {
        data.datatype = MNIST_DATATYPE_NONE;
        data.width = 0u;
        return false;
    }
    // calculate data memory footprint per sample
    uint32_t memory_footprint = 1;
    for (std::uint32_t dimension_size : data.dimension_sizes)
    {
        memory_footprint *= dimension_size;
    }
    memory_footprint *= data.width;
    // extract data
    for (uint32_t i = 0; i < data.sample_count; i++)
    {
        auto *sample = new uint8_t[memory_footprint];
        if (!(read_ifstream_safe(file, reinterpret_cast<char *>(sample), memory_footprint)))
            return false;
        data.data.push_back(sample);
    }
    return true;
}

