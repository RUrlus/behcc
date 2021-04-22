/*
    include/behcc/hash_encoder.h: Defines HashBinaryEncoder

    Copyright (c) 2021 Ralph Urlus <rurlus.dev@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

*/

#ifndef INCLUDE_BEHCC_HASH_ENCODER_H_
#define INCLUDE_BEHCC_HASH_ENCODER_H_

#include <behcc/behcc.h>
#include <behcc/numpy.h>
#include <behcc/common.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <numpy/arrayobject.h>

namespace py = pybind11;


namespace behcc {
namespace pyapi {

class HashBinaryEncoder;
void bind_hashbinaryencoding(py::module &m);

}  // namespace pyapi
}  // namespace behcc
#endif  // INCLUDE_BEHCC_HASH_ENCODER_H_
