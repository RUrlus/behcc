/*
    src/behcc_python.cpp: Python API for behcc.h

    Copyright (c) 2021 Ralph Urlus <rurlus.dev@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

*/

#ifndef INCLUDE_BEHCC_C_H_
#define INCLUDE_BEHCC_C_H_
#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <iostream>

namespace py = pybind11;


namespace behcc {

static inline bool is_aligned(PyArrayObject* src) {
    return PyArray_CHKFLAGS(src, NPY_ARRAY_ALIGNED);
}

static inline bool is_aligned(PyObject* src) {
    return PyArray_CHKFLAGS(
        reinterpret_cast<PyArrayObject*>(src),
        NPY_ARRAY_ALIGNED
    );
}

static inline char get_dtype(PyObject* src) {
    auto arr = reinterpret_cast<PyArrayObject*>(src);
    PyArray_Descr * descr = PyArray_DTYPE(arr);
    Py_INCREF(descr);
    return descr->type;
}

static inline char* get_ptr(PyArrayObject* arr) {
    return reinterpret_cast<char*>(PyArray_DATA(arr));
}

static inline char* get_ptr(PyObject* src) {
    return reinterpret_cast<char*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(src)));
}

static inline int get_itemsize(PyObject* src) {
    return PyArray_ITEMSIZE(reinterpret_cast<PyArrayObject*>(src));
}

static inline int get_itemsize(PyArrayObject* arr) {
    return PyArray_ITEMSIZE(arr);
}

static inline size_t get_size(PyObject* src) {
    auto arr = reinterpret_cast<PyArrayObject*>(src);
    npy_intp* ptr = PyArray_DIMS(arr);
    size_t ndim = PyArray_NDIM(arr);
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= ptr[i];
    }
    return size;
}

static inline size_t get_size(PyArrayObject* arr) {
    npy_intp* ptr = PyArray_DIMS(arr);
    size_t ndim = PyArray_NDIM(arr);
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= ptr[i];
    }
    return size;
}

static inline size_t get_stride(PyObject* src) {
    auto arr = reinterpret_cast<PyArrayObject*>(src);
    npy_intp* ptr = PyArray_STRIDES(arr);
    size_t ndim = PyArray_NDIM(arr);
    if (ndim == 1) {
        return ptr[0];
    }
    npy_intp* end = reinterpret_cast<npy_intp*>(
        reinterpret_cast<char *>(ptr) +  ndim
    );
    return *std::min_element(ptr, end);
}

static inline size_t get_stride(PyArrayObject* arr) {
    npy_intp* ptr = PyArray_STRIDES(arr);
    size_t ndim = PyArray_NDIM(arr);
    if (ndim == 1) {
        return ptr[0];
    }
    npy_intp* end = reinterpret_cast<npy_intp*>(
        reinterpret_cast<char *>(ptr) +  ndim
    );
    return *std::min_element(ptr, end);
}

}  // namespace behcc
#endif  // INCLUDE_BEHCC_C_H_
