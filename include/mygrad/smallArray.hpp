#pragma once

#include <array>
#include <string>
#include <algorithm>
#include <initializer_list>

template <typename T, T capacity_>
struct SmallArray {
    std::array<T, capacity_> arr;
    T size_;

public:
    SmallArray(std::initializer_list<T> initList) : 
        arr{}, size_(initList.size()) {
            if (size_ > capacity_) {
                throw std::runtime_error (
                    "array size must not exceed the capacity_: " + 
                    std::to_string(initList.size()) + " > " + std::to_string(capacity_));
            }
            std::copy(initList.begin(), initList.end(), arr.begin());
        }

    SmallArray( const SmallArray& other ) = default;
    SmallArray( SmallArray&& other) = default;

    SmallArray& operator=( const SmallArray& other ) = default;
    SmallArray& operator=( SmallArray&& other ) = default;
    SmallArray& operator=( std::initializer_list<T> initList ) {
        if (initList.size() > capacity_) throw std::runtime_error("trying to assign data of a size too big to a small array");
        std::copy(initList.begin(), initList.end(), arr.begin());
        size_ = initList.size();
        return *this;
    }


    inline constexpr T size() const noexcept { return size_; }
    inline constexpr T capacity() const noexcept { return capacity_; }
    inline constexpr T operator[](size_t i) const noexcept {return arr[i]; }
    inline constexpr T& operator[](size_t i) noexcept {return arr[i]; }
    inline constexpr T at(size_t i) const {
        if (i < 0 or i >= size_) throw std::out_of_range(std::to_string(i) + " is out of range for SmallArray");
        return arr[i];
    }
    inline constexpr T& at(size_t i) {
        if (i < 0 or i >= size_) throw std::out_of_range(std::to_string(i) + " is out of range for SmallArray");
        return arr[i];
    }

    inline void print() const noexcept {
        std::cout << "{";
        for (size_t i = 0; i < size_; i++) {
            std::cout << arr[i];
            if (i != size_ - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}\n";
    }
};