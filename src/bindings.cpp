/*
    src/bindings.cpp: Python bindings for behcc.h

    Copyright (c) 2021 Ralph Urlus <rurlus.dev@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

*/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <behcc_python.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(EXTENSION_MODULE_NAME, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: behcc_extension

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    behcc::pyapi::bind_hashbinaryencoding(m);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
