//
// Created by Boyan Atanasov on 12/01/2020.
//
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>

#include "mnist.h"

static bool open_ifstream(std::ifstream &file, const std::string &filepath) noexcept;

static bool read_ifstream(std::ifstream &file, char *destination, std::size_t size) noexcept;

static std::uint32_t get_dimensions_from_magic(const std::uint32_t &magic) noexcept;

static std::uint32_t get_datatype_from_magic(const std::uint32_t &magic) noexcept;

void correct_if_lsb(std::uint32_t *&val) noexcept;

const static std::map<mnist::mnist_datatype, uint8_t> datatype_widths = {
        {mnist::MNIST_UNSIGNED_BYTE, 1},
        {mnist::MNIST_SIGNED_BYTE,   1},
        {mnist::MNIST_SHORT,         2},
        {mnist::MNIST_INT,           4},
        {mnist::MNIST_FLOAT,         4},
        {mnist::MNIST_DOUBLE,        8}
};

const static std::map<mnist::mnist_datatype, std::string> datatype_names = {
        {mnist::MNIST_UNSIGNED_BYTE, "Unsigned Byte"},
        {mnist::MNIST_SIGNED_BYTE,   "Signed Byte"},
        {mnist::MNIST_SHORT,         "Short Integer"},
        {mnist::MNIST_INT,           "Integer"},
        {mnist::MNIST_FLOAT,         "Float"},
        {mnist::MNIST_DOUBLE,        "Double"}
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

static bool open_ifstream(std::ifstream &file, const std::string &filepath) noexcept
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
        std::cerr << "Could not open file at \"" << filepath << "\"" << std::endl;
        std::cerr << e.what();
        open = false;
    }
    return open;
}

static bool read_ifstream(std::ifstream &file, char *destination, std::size_t size) noexcept
{
    bool read = true;
    auto tell = file.tellg();
    try
    {
        file.read(destination, size);
    } catch (std::exception &e)
    {
        std::cerr << "Could not read stream at byte " << tell << std::endl;
        std::cerr << e.what();
        read = false;
    }
    return read;
}

static std::uint32_t get_dimensions_from_magic(const std::uint32_t &magic) noexcept
{
    constexpr static std::uint32_t dimension_mask = 0x000000FF;
    // sample_size is always first dimension so we need
    // to subtract one from total count
    std::uint32_t dimension_count = (magic & dimension_mask) - 1U;
    return dimension_count;
}

static std::uint32_t get_datatype_from_magic(const std::uint32_t &magic) noexcept
{
    constexpr static std::uint32_t datatype_mask = 0x0000000F; // extract datatype
    // maximum datatype identifier is 0xE
    std::uint32_t datatype = (magic >> 8u) & datatype_mask;
    return datatype;
}

void correct_if_lsb(std::uint32_t *val) noexcept
{
    if (mnist::util::architecture_is_lsb())
    {
        *val = mnist::util::revb_uint32(*val);
    }
}

bool mnist::dataset_load(mnist_data &data, const std::string &filepath)
{
    std::ifstream file;
    std::uint32_t dimension_count;
    std::uint32_t dimension_size;
    std::uint32_t datatype;
    std::uint32_t memory_footprint;
    std::uint32_t i;
    // open file @ filepath if possible, if not, return
    if (!open_ifstream(file, filepath))
    { return false; }
    else
    { data.filepath_history = filepath; }
    // read magic
    if (!(read_ifstream(file, reinterpret_cast<char *>(&data.magic),
                        sizeof(std::uint32_t))))
    { return false; }
    // read sample count
    if (!(read_ifstream(file, reinterpret_cast<char *>(&data.sample_count),
                        sizeof(std::uint32_t))))
    { return false; }
    // correct endianness in magic number if needed
    correct_if_lsb(&data.magic);
    correct_if_lsb(&data.sample_count);
    dimension_count = get_dimensions_from_magic(data.magic);
    for (i = 0; i < dimension_count; i++)
    {
        if (!(read_ifstream(file, reinterpret_cast<char *>(&dimension_size),
                            sizeof(std::uint32_t))))
        { return false; }
        correct_if_lsb(&dimension_size);
        data.dimension_sizes.push_back(dimension_size);
    }
    datatype = get_datatype_from_magic(data.magic);
    if (datatype_widths.find(static_cast<mnist_datatype>(datatype)) != datatype_widths.end())
    {
        data.datatype = static_cast<mnist_datatype>(datatype);
        data.width = datatype_widths.at(static_cast<mnist_datatype>(datatype));
    } else
    {
        data.datatype = MNIST_DATATYPE_NONE;
        data.width = 0u;
        return false;
    }
    // calculate data memory footprint per sample
    memory_footprint = 1;
    for (auto ds : data.dimension_sizes)
    {
        memory_footprint *= ds;
    }
    memory_footprint *= data.width;
    // extract binary data from file
    for (i = 0; i < data.sample_count; i++)
    {
        auto *sample = new uint8_t[memory_footprint];
        if (!(read_ifstream(file, reinterpret_cast<char *>(sample), memory_footprint)))
        { return false; }
        data.data.push_back(sample);
    }
    return true;
}

void mnist::dataset_info(mnist::mnist_data &data)
{
    std::stringstream ss;
    std::uint32_t inner_data_size = 1;
    ss << "{\t" << std::endl;
    ss << "\t" << "Dataset filepath: \"" << data.filepath_history << std::endl;
    ss << "\t" << "File magic: " << std::hex << data.magic << std::dec <<  std::endl;
    ss << "\t" << "Fundamental data width: " << static_cast<std::uint32_t>(data.width) << std::endl;
    ss << "\t" << "Fundamental data type: " << datatype_names.at(data.datatype) << std::endl;
    ss << "\t" << "Sample count: " << data.sample_count << std::endl;
    if(!data.dimension_sizes.empty())
    {
        ss << "\t" << "Inner data dimensions: ";
        for(auto& dimension : data.dimension_sizes)
        {
            ss << "[" << dimension << "]";
            inner_data_size *= dimension;
        }
        ss << std::endl;
    }
    ss << "\t" << "Data: {... " << static_cast<std::uint32_t>(data.width) * inner_data_size
                            * data.sample_count<< " bytes ...}" << std::endl;
    ss << "}\t" << std::endl;
    std::cout << ss.str();
}
