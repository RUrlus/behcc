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
    bool param_check = true;
    int itemsize;
    size_t size;
    size_t stride;
    size_t encoding_size;
    size_t i_encoding_size;
    int64_t seed;
    uint64_t i_seed;
    int* result_ptr = nullptr;
    char * data;
    PyArrayObject* arr;

    void hash8bit_enc() {
        char* ptr = data;
        for (int i = 0; i < size; i++) {
            hash_to_binary_array_8bits(ptr, itemsize, result_ptr, i_seed);
            ptr += stride;
            result_ptr += i_encoding_size;
        }
    } // 8bit_enc

    void hash16bit_enc() {
        char* ptr = data;
        for (int i = 0; i < size; i++) {
            hash_to_binary_array_16bits(ptr, itemsize, result_ptr, i_seed);
            ptr += stride;
            result_ptr += i_encoding_size;
        }
    } // 16bit_enc

    void hash32bit_enc() {
        char* ptr = data;
        for (int i = 0; i < size; i++) {
            hash_to_binary_array_16bits(ptr, itemsize, result_ptr, i_seed);
            ptr += stride;
            result_ptr += i_encoding_size;
        }
    } // 32bit_enc

    void hash64bit_enc() {
        char* ptr = data;
        for (int i = 0; i < size; i++) {
            hash_to_binary_array_64bits(ptr, itemsize, result_ptr, i_seed);
            ptr += stride;
            result_ptr += i_encoding_size;
        }
    } // 64bit_enc

    void hash128bit_enc() {
        char* ptr = data;
        for (int i = 0; i < size; i++) {
            hash_to_binary_array_128bits(ptr, itemsize, result_ptr, i_seed);
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
    py::object encoding_size_ = py::none();
    py::object seed_ = py::none();

    explicit HashBinaryEncoder(size_t encoding_size, int64_t seed) :
        seed{seed}, encoding_size{encoding_size} {}

    void set_params(py::kwargs kwargs) {
        if (kwargs.contains("encoding_size")) {
            encoding_size = kwargs["encoding_size"].cast<size_t>();
        }
        if (kwargs.contains("seed")) {
            seed = kwargs["seed"].cast<uint64_t>();
        }
    }

    py::dict get_params() {
        using pybind11::literals::operator""_a;
        return py::dict(
            "seed"_a = seed,
            "encoding_size"_a = encoding_size
        );
    }

    void fit(const py::object& src) {
        if (encoding_size > 128) {
            throw encoding_size_error("Maximum encoding_size is 128");
        } else if (encoding_size < 1) {
            throw encoding_size_error("Minimum encoding_size is 1");
        } else {
            encoding_size_ = py::cast(encoding_size);
            i_encoding_size = encoding_size;
        }
        if (seed >= 0) {
            i_seed = static_cast<uint64_t>(seed);
            seed_ = py::cast(seed);
        } else {
            throw uninitialiazed_error("seed must be integer >= 0");
        }
        param_check = false;
    }

    py::array_t<int> transform(const py::object& src) {
        if (param_check) {
            fit(src);
        }
        arr = reinterpret_cast<PyArrayObject*>(src.ptr());
        if (!is_aligned(arr)) {
            throw not_aligned_error("Unaligned arrays are not supoorted");
        }
        data = get_ptr(arr);
        size = get_size(arr);
        stride = get_stride(arr);
        itemsize = get_itemsize(arr);

        auto result = py::array_t<int>(i_encoding_size * size);
        result_ptr = reinterpret_cast<int*>(result.request().ptr);

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

    py::array_t<int> fit_transform(py::object src) {
        fit(src);
        return transform(src);
    }

    py::dict _more_tags() {
        using pybind11::literals::operator""_a;
        auto list = py::list(3);
        list.append<std::string>("1darray");
        list.append<std::string>("1darray");
        list.append<std::string>("categorical");
        return py::dict(
            "allow_nan"_a = false,
            "stateless"_a = true,
            "X_types"_a = list
        );
    }
};

void bind_hashbinaryencoding(py::module &m) {
    py::class_<HashBinaryEncoder>(
        m,
        "HashBinaryEncoder",
        R"pbdoc(
            HashBinaryEncoder encodes the bits of the hashed inputs
            as an integer array.

            The eucledian distance between any encoding vectors
            follows an approximate Normal distribution.
        )pbdoc"
    )
        .def_readonly("n_components_", &HashBinaryEncoder::encoding_size_)
        .def_readonly("seed_", &HashBinaryEncoder::seed_)
        .def(py::init<size_t, int64_t>(), R"pbdoc(
            Initialise StringBinaryEncoder.

            Parameters
            ----------
            encoding_size : {1...128}
                size of the encoding, must be between 1 and 128 inclusive.
                encoding of sizes {8, 16, 32, 64, 128} are relatively faster
                to compute than variable sized encodings
            seed : int, optional
                seed used for when computing the hash, if None or 0 a seed
                will be generated
        )pbdoc",
        py::arg("encoding_size") = 64,
        py::arg("seed") = 0
        )
        .def(
            "get_params",
            &HashBinaryEncoder::get_params,
            R"pbdoc(
            Set parameters.

            Returns
            -------
            dict(str: value)
                parameters set
            )pbdoc"
        )
        .def(
            "set_params",
            &HashBinaryEncoder::set_params,
            R"pbdoc(
            Set parameters.

            Parameters
            ----------
            **kwargs
            )pbdoc"
        )
        .def(
            "fit",
            &HashBinaryEncoder::fit,
            R"pbdoc(
            Fit encoding.

            Parameters
            ----------
            src : np.ndarray
                numpy array to be hashed and encoded

            )pbdoc",
            py::arg("src") = py::none()
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

            Returns
            -------
            np.ndarray[int32]
                array containing the encoding
            )pbdoc",
            py::arg("src")
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
            py::arg("src")
        );
}

}  // namespace pyapi
}  // namespace behcc
