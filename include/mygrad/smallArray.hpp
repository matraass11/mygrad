#pragma once

#include <array>
#include <string>
#include <algorithm>
#include <initializer_list>

namespace mygrad {

template <typename T, T capacity_>
struct SmallArray {
    std::array<T, capacity_> data;
    T size_;

public:
    SmallArray(std::initializer_list<T> initList) : 
        data{}, size_(initList.size()) {
            if (size_ > capacity_) {
                throw std::runtime_error (
                    "array size must not exceed the capacity_: " + 
                    std::to_string(initList.size()) + " > " + std::to_string(capacity_));
            }
            std::copy(initList.begin(), initList.end(), data.begin());
        }

    SmallArray(size_t size_) : 
    data{}, size_(size_) {
        if (size_ > capacity_) {
            throw std::runtime_error (
                "array size must not exceed the capacity_: " + 
                std::to_string(size_) + " > " + std::to_string(capacity_));
        }
    };

    SmallArray( const SmallArray& other ) = default;
    SmallArray( SmallArray&& other) = default;

    SmallArray& operator=( const SmallArray& other ) = default;
    SmallArray& operator=( SmallArray&& other ) = default;
    SmallArray& operator=( std::initializer_list<T> initList ) {
        if (initList.size() > capacity_) throw std::runtime_error("trying to assign data of a size too big to a small array");
        std::copy(initList.begin(), initList.end(), data.begin());
        size_ = initList.size();
        return *this;
    }


    inline constexpr T* begin() noexcept { return &data[0]; }
    inline constexpr T* end() noexcept { return &data[size_ - 1]; }
    inline constexpr const T* begin() const noexcept { return &data[0] ; }
    inline constexpr const T* end() const noexcept { return &data[size_ - 1]; }

    inline constexpr T size() const noexcept { return size_; }
    inline constexpr T capacity() const noexcept { return capacity_; }
    inline constexpr T operator[](size_t i) const noexcept { return data[i]; }
    inline constexpr T& operator[](size_t i) noexcept { return data[i]; }
    inline constexpr T at(size_t i) const {
        if (i < 0 or i >= size_) throw std::out_of_range(std::to_string(i) + " is out of range for SmallArray");
        return data[i];
    }
    inline constexpr T& at(size_t i) {
        if (i < 0 or i >= size_) throw std::out_of_range(std::to_string(i) + " is out of range for SmallArray");
        return data[i];
    }

    inline void print() const noexcept {
        std::cout << "{";
        for (size_t i = 0; i < size_; i++) {
            std::cout << data[i];
            if (i != size_ - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}\n";
    }


    bool operator==(const SmallArray<T, capacity_>& other) const {
        return true ? (data == other.data and size_ == other.size_) : false;
    }

    bool operator!=(const SmallArray<T, capacity_>& other) const {
        return !(*this == other);
    }
};

template<typename T, size_t capacity>
inline std::ostream& operator<<(std::ostream& out, const SmallArray<T, capacity>& v) {
    out << "{";
    for (size_t i = 0; i < v.size(); i++) {
        out << v[i];
        if (i != v.size() - 1) {
            out << ", ";
        }
    }
    out << "}\n";
    return out;
}

} // namespace mygrad
