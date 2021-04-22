/*
    include/behcc/common.h: Common functionality

    Copyright (c) 2021 Ralph Urlus <rurlus.dev@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

*/

#ifndef INCLUDE_BEHCC_COMMON_H_
#define INCLUDE_BEHCC_COMMON_H_

#include <exception>

namespace behcc {
namespace pyapi {

struct not_aligned_error : std::exception {
    const char* p_message;
    explicit not_aligned_error(const char* message) : p_message(message) {}
    const char* what() const throw() { return p_message; }
};

struct encoding_size_error : std::exception {
    const char* p_message;
    explicit encoding_size_error(const char* message) : p_message(message) {}
    const char* what() const throw() { return p_message; }
};

struct uninitialiazed_error : std::exception {
    const char* p_message;
    explicit uninitialiazed_error(const char* message) : p_message(message) {}
    const char* what() const throw() { return p_message; }
};

}  // namespace pyapi
}  // namespace behcc

#endif  // INCLUDE_BEHCC_COMMON_H_
