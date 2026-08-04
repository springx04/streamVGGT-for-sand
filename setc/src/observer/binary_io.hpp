#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace omnivggt::observer {

// The observer stores and transports all scalar values in explicit little-endian
// form.  This keeps the history files portable between Windows and Linux and
// avoids relying on compiler struct packing.
class BinaryWriter {
public:
    void u8(const std::uint8_t value) { data_.push_back(value); }

    void u16(const std::uint16_t value) {
        data_.push_back(static_cast<std::uint8_t>(value & 0xffU));
        data_.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
    }

    void u32(const std::uint32_t value) {
        for (unsigned int shift = 0; shift < 32U; shift += 8U) {
            data_.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
        }
    }

    void u64(const std::uint64_t value) {
        for (unsigned int shift = 0; shift < 64U; shift += 8U) {
            data_.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
        }
    }

    void f32(const float value) {
        static_assert(sizeof(float) == sizeof(std::uint32_t), "float must be 32-bit");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        u32(bits);
    }

    void boolean(const bool value) { u8(value ? 1U : 0U); }

    void string(const std::string& value) {
        if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("string is too large to serialize");
        }
        u32(static_cast<std::uint32_t>(value.size()));
        bytes(value.data(), value.size());
    }

    void bytes(const void* data, const std::size_t size) {
        if (size == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("cannot serialize null bytes");
        }
        const auto* first = static_cast<const std::uint8_t*>(data);
        data_.insert(data_.end(), first, first + size);
    }

    void bytes(const std::vector<std::uint8_t>& value) { bytes(value.data(), value.size()); }

    const std::vector<std::uint8_t>& data() const noexcept { return data_; }
    std::vector<std::uint8_t>& data() noexcept { return data_; }

private:
    std::vector<std::uint8_t> data_;
};

class BinaryReader {
public:
    explicit BinaryReader(const std::vector<std::uint8_t>& data) : data_(data.data()), size_(data.size()) {}
    BinaryReader(const std::uint8_t* data, const std::size_t size) : data_(data), size_(size) {}

    std::uint8_t u8() {
        require(1U);
        return data_[offset_++];
    }

    std::uint16_t u16() {
        require(2U);
        const std::uint16_t value = static_cast<std::uint16_t>(data_[offset_])
            | static_cast<std::uint16_t>(data_[offset_ + 1U] << 8U);
        offset_ += 2U;
        return value;
    }

    std::uint32_t u32() {
        require(4U);
        std::uint32_t value = 0;
        for (unsigned int shift = 0; shift < 32U; shift += 8U) {
            value |= static_cast<std::uint32_t>(data_[offset_++]) << shift;
        }
        return value;
    }

    std::uint64_t u64() {
        require(8U);
        std::uint64_t value = 0;
        for (unsigned int shift = 0; shift < 64U; shift += 8U) {
            value |= static_cast<std::uint64_t>(data_[offset_++]) << shift;
        }
        return value;
    }

    float f32() {
        const std::uint32_t bits = u32();
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    bool boolean() { return u8() != 0U; }

    std::string string() {
        const std::uint32_t length = u32();
        require(length);
        std::string value(reinterpret_cast<const char*>(data_ + offset_), length);
        offset_ += length;
        return value;
    }

    std::vector<std::uint8_t> bytes(const std::size_t length) {
        require(length);
        std::vector<std::uint8_t> value(data_ + offset_, data_ + offset_ + length);
        offset_ += length;
        return value;
    }

    const std::uint8_t* view(const std::size_t length) {
        require(length);
        const auto* result = data_ + offset_;
        offset_ += length;
        return result;
    }

    std::size_t remaining() const noexcept { return size_ - offset_; }
    std::size_t offset() const noexcept { return offset_; }

private:
    void require(const std::size_t count) const {
        if (count > size_ - offset_) {
            throw std::runtime_error("truncated binary data");
        }
    }

    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t offset_ = 0;
};

}  // namespace omnivggt::observer
