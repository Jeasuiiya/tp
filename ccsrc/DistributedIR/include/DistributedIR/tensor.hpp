#pragma once

#ifndef FRAMEWORK_IR_TENSOR_H
#define FRAMEWORK_IR_TENSOR_H

#include <cstdint>
#include <numeric>
#include <vector>

#include "common/error.hpp"
#include "common/fmt.hpp"
#include "common/result_macro.hpp"
namespace framework {

/**
 * DataType
 * from jaxlib/xla_client.py
 */
enum class DataType {
    BOOL,      // bool
    U8,        // uint8
    I8,        // int8
    U16,       // uint16
    I16,       // int16
    U32,       // uint32
    I32,       // int32
    U64,       // uint64
    I64,       // int64
    F8E4M3FN,  // numpy float8_e4m3fn
    F8E5M2,    // numpy float8_e5m2
    BF16,      // bfloat16
    F16,       // float16
    F32,       // float32
    F64,       // float64
    Other      // record in python
    // todo(yiguangzheng): support these type
    // C64,
    // C128,
    // TUPLE,
    // TOKEN,
};
using shape_t = std::vector<ssize_t>;
struct AbstractTensor {
    friend struct fmt::formatter<framework::AbstractTensor>;

    //   protected:
    DataType dtype;
    // if size of shape is 0, represents a scalar
    shape_t shape;
    size_t size{0};

    void* data = nullptr;
    // byte length
    size_t length{0};

    //   public:
    AbstractTensor(const AbstractTensor& t) : dtype(t.dtype), shape(t.shape), size(t.size), length(t.length) {
        data = std::malloc(length);
        memcpy(data, t.data, length);
    }
    AbstractTensor& operator=(const AbstractTensor& t) {
        if (&t == this) {
            return *this;
        }
        dtype = t.dtype;
        shape = t.shape;
        size = t.size;
        length = t.length;
        data = std::malloc(length);
        memcpy(data, t.data, length);
        return *this;
    }
    AbstractTensor(AbstractTensor&& t) noexcept
        : dtype(t.dtype), shape(std::move(t.shape)), size(t.size), data(t.data), length(t.length) {
        t.data = nullptr;
    }
    AbstractTensor& operator=(AbstractTensor&& t) noexcept {
        if (&t == this) {
            return *this;
        }
        dtype = t.dtype;
        shape = t.shape;
        size = t.size;
        length = t.length;
        data = t.data;
        t.data = nullptr;
        return *this;
    }

    AbstractTensor(DataType dtype, const shape_t& shape) {
        this->dtype = dtype;
        this->shape = shape;
        size = std::accumulate(this->shape.begin(), this->shape.end(), 1,
                               [](const auto& a, const auto& b) { return a * b; });
    }
    // Tensor require ptr's ownership, auto free memory
    AbstractTensor(DataType dtype, const shape_t& shape, void* ptr, size_t length) : AbstractTensor(dtype, shape) {
        this->length = length;
        data = ptr;
    };
    ~AbstractTensor() {
        Release();
    }
    virtual bool operator==(const AbstractTensor& t) {
        return shape == t.shape && dtype == t.dtype && data == t.data;
    }

    template <class T>
    T* Cast() {
        return static_cast<T*>(data);
    }

    template <class T>
    void Alloc() {
        length = sizeof(T) * size;
        data = std::aligned_alloc(64, length);
    }
    void Release() {
        if (data != nullptr) {
            free(data);
            data = nullptr;
        }
    }
    // Don't release this ptr manually
    template <class T>
    cpp::result<T&, Error> At(size_t index) {
        if (length % sizeof(T) != 0 || index >= length / sizeof(T) || data == nullptr) {
            return cpp::fail(Error(Kind::Invalid, "cannot cast as T or index out of range or data is null"));
        }
        return *(Cast<T>() + index);
    }
};

}  // namespace framework

// NOLINTBEGIN(readability-identifier-naming)
template <>
struct fmt::formatter<framework::DataType> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const framework::DataType& t, FormatContext& ctx) const -> decltype(ctx.out()) {
        std::string s;
        switch (t) {
            case framework::DataType::BOOL:
                s = "BOOL";
                break;
            case framework::DataType::U8:
                s = "U8";
                break;
            case framework::DataType::I8:
                s = "I8";
                break;
            case framework::DataType::U16:
                s = "U16";
                break;
            case framework::DataType::I16:
                s = "I16";
                break;
            case framework::DataType::U32:
                s = "U32";
                break;
            case framework::DataType::I32:
                s = "I32";
                break;
            case framework::DataType::U64:
                s = "U64";
                break;
            case framework::DataType::I64:
                s = "I64";
                break;
            case framework::DataType::F8E4M3FN:
                s = "F8E4M3FN";
                break;
            case framework::DataType::F8E5M2:
                s = "F8E5M2";
                break;
            case framework::DataType::BF16:
                s = "BF16";
                break;
            case framework::DataType::F16:
                s = "F16";
                break;
            case framework::DataType::F32:
                s = "F32";
                break;
            case framework::DataType::F64:
                s = "F64";
                break;
            default:
                break;
        }
        return fmt::format_to(ctx.out(), "{}", s);
    }
};
template <>
struct fmt::formatter<framework::AbstractTensor> : fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const framework::AbstractTensor& t, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "AbstractTensor(dtype={}, shape={})", t.dtype, t.shape);
    }
};
// NOLINTEND(readability-identifier-naming)
#endif
