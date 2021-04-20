/*
    src/behcc/hash_encoder.cpp: Defines HashBinaryEncoder

    Copyright (c) 2021 Ralph Urlus <rurlus.dev@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

*/
#include <behcc/hash_encoder.h>

namespace py = pybind11;

namespace behcc { namespace pyapi {

class HashBinaryEncoder {

 protected:
    int itemsize;
    size_t size;
    size_t stride;
    size_t i_encoding_size;
    uint64_t seed;
    char * data;
    PyArrayObject* arr;
    int* result_ptr = nullptr;

    void hash8bit_enc() {
        char* ptr = data;
        for (int i = 0; i < size; i++) {
            hash_to_binary_array_8bits(ptr, itemsize, result_ptr, seed);
            ptr += stride;
            result_ptr += i_encoding_size;
        }
    } // 8bit_enc

    void hash16bit_enc() {
        char* ptr = data;
        for (int i = 0; i < size; i++) {
            hash_to_binary_array_16bits(ptr, itemsize, result_ptr, seed);
            ptr += stride;
            result_ptr += i_encoding_size;
        }
    } // 16bit_enc

    void hash32bit_enc() {
        char* ptr = data;
        for (int i = 0; i < size; i++) {
            hash_to_binary_array_16bits(ptr, itemsize, result_ptr, seed);
            ptr += stride;
            result_ptr += i_encoding_size;
        }
    } // 32bit_enc

    void hash64bit_enc() {
        char* ptr = data;
        for (int i = 0; i < size; i++) {
            hash_to_binary_array_64bits(ptr, itemsize, result_ptr, seed);
            ptr += stride;
            result_ptr += i_encoding_size;
        }
    } // 64bit_enc

    void hash128bit_enc() {
        char* ptr = data;
        for (int i = 0; i < size; i++) {
            hash_to_binary_array_128bits(ptr, itemsize, result_ptr, seed);
            ptr += stride;
            result_ptr += i_encoding_size;
        }
    } // 64bit_enc

    void hash_dyn_bit_enc() {
        char* ptr = data;
        if (i_encoding_size > 64) {
            for (int i = 0; i < size; i++) {
                hash_to_binary_array_128bits(ptr, itemsize, i_encoding_size, result_ptr);
                ptr += stride;
                result_ptr += i_encoding_size;
            }
        } else {
            for (int i = 0; i < size; i++) {
                hash_to_binary_array_64bits(ptr, itemsize, i_encoding_size, result_ptr);
                ptr += stride;
                result_ptr += i_encoding_size;
            }
        }
    } // dyn_bit_enc

 public:
    HashBinaryEncoder() {}

    void fit(py::object src, size_t encoding_size) {
        arr = reinterpret_cast<PyArrayObject*>(src.ptr());

        if (!is_aligned(arr)) {
            throw not_aligned_error("Unaligned arrays are not supoorted");
        }

        if (encoding_size > 128) {
            throw encoding_size_error("Maximum encoding_size is 128");
        } else if (encoding_size < 1) {
            throw encoding_size_error("Minimum encoding_size is 1");
        } else {
            i_encoding_size = encoding_size;
        }

        data = get_ptr(arr);
        size = get_size(arr);
        stride = get_stride(arr);
        itemsize = get_itemsize(arr);
    }

    py::array_t<int> transform(py::object src, uint64_t seed) {
        auto result = py::array_t<int>(i_encoding_size * size);
        result_ptr = reinterpret_cast<int*>(result.request().ptr);
        seed = seed;

        switch (i_encoding_size) {
            case 8:
                hash8bit_enc();
                break;
            case 16:
                hash16bit_enc();
                break;
            case 32:
                hash32bit_enc();
                break;
            case 64:
                hash64bit_enc();
                break;
            case 128:
                hash128bit_enc();
                break;
            default:
                hash_dyn_bit_enc();
                break;
        }
        result.resize({size, i_encoding_size});
        result_ptr = nullptr;
        return result;
    }

    py::array_t<int> fit_transform(py::object src, size_t encoding_size, uint64_t seed) {
        fit(src, encoding_size);
        return transform(src, seed);
    }
};

void bind_hashbinaryencoding(py::module &m) {
    py::class_<HashBinaryEncoder>(m, "HashBinaryEncoder")
        .def(py::init<>(), R"pbdoc(
            Initialise StringBinaryEncoder.
        )pbdoc")
        .def(
            "fit",
            &HashBinaryEncoder::fit,
            R"pbdoc(
            Set input and parameters.

            Parameters
            ----------
            src : np.ndarray
                numpy array to be hashed and encoded
            encoding_size : {1...128}
                size of the encoding, must be between 1 and 128 inclusive.
                encoding of sizes {8, 16, 32, 64, 128} are relatively faster
                to compute than variable sized encodings

            )pbdoc",
            py::arg("src"),
            py::arg("encoding_size")
        )
        .def(
            "transform",
            &HashBinaryEncoder::transform,
            R"pbdoc(
            Create binary encoding of hash of the input.

            Parameters
            ----------
            src : np.ndarray
                numpy array to be hashed and encoded
            seed : int, optional
                seed used for when computing the hash,
                default is 42

            Returns
            -------
            np.ndarray[int32]
                array containing the encoding
            )pbdoc",
            py::arg("src"),
            py::arg("seed") = 42
        )
        .def(
            "fit_transform",
            &HashBinaryEncoder::fit_transform,
            R"pbdoc(
            Create binary encoding of hash of the input.

            Parameters
            ----------
            src : np.ndarray
                numpy array to be hashed and encoded
            encoding_size : {1...128}
                size of the encoding, must be between 1 and 128 inclusive.
                encoding of sizes {8, 16, 32, 64, 128} are relatively faster
                to compute than variable sized encodings
            seed : int, optional
                seed used for when computing the hash,
                default is 42

            Returns
            -------
            np.ndarray[int32]
                array containing the encoding

            )pbdoc",
            py::arg("src"),
            py::arg("encoding_size"),
            py::arg("seed") = 42
        );
}

}  // namespace pyapi
}  // namespace behcc
