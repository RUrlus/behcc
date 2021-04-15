/*
    src/behcc.cpp: Binary Encoding for High Cardinality Categoricals

    Copyright (c) 2021 Ralph Urlus <rurlus.dev@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

*/
#include <cmath>
#include <string>
#include <algorithm>
#include "../extern/xxHash/xxhash.h"

#ifndef INCLUDE_BEHCC_HPP_
#define INCLUDE_BEHCC_HPP_

/*
###################################################################
#                    Private utility functions                    #
###################################################################
*/

template <typename T>
inline void p_btbe(T input, const int nbits, int* result) {
#if defined(IS_BIG_ENDIAN)
    // first bit of variable is store at first location in memory
    for (int b = 0; b < nbits; b++) {
        result[b] = ((input >> b) & 1);
#else
    // first bit of variable is store at last location in memory
    int j = 0;
    for (int b = nbits-1; b != -1; b--) {
        result[j] = ((input >> b) & 1);
        j++;
#endif
    }
}

template <typename T>
inline void p_btbe_8bit(T input, int* result) {
#if defined(IS_BIG_ENDIAN)
    // first bit of variable is store at first location in memory
    for (int b = 0; b < 8; b++) {
        result[b] = ((input >> b) & 1);
#else
    // first bit of variable is store at last location in memory
    int j = 0;
    for (int b = 7; b != -1; b--) {
        result[j] = ((input >> b) & 1);
        j++;
#endif
    }
}

template <typename T>
inline void p_btbe_16bit(T input, int* result) {
#if defined(IS_BIG_ENDIAN)
    // first bit of variable is store at first location in memory
    for (int b = 0; b < 16; b++) {
        result[b] = ((input >> b) & 1);
#else
    // first bit of variable is store at last location in memory
    int j = 0;
    for (int b = 15; b != -1; b--) {
        result[j] = ((input >> b) & 1);
        j++;
#endif
    }
}

template <typename T>
inline void p_btbe_32bit(T input, int* result) {
#if defined(IS_BIG_ENDIAN)
    // first bit of variable is store at first location in memory
    for (int b = 0; b < 32; b++) {
        result[b] = ((input >> b) & 1);
#else
    // first bit of variable is store at last location in memory
    int j = 0;
    for (int b = 31; b != -1; b--) {
        result[j] = ((input >> b) & 1);
        j++;
#endif
    }
}

template <typename T>
inline void p_btbe_64bit(T input, int* result) {
#if defined(IS_BIG_ENDIAN)
    // first bit of variable is store at first location in memory
    for (int b = 0; b < 64; b++) {
        result[b] = ((input >> b) & 1);
#else
    // first bit of variable is store at last location in memory
    int j = 0;
    for (int b = 63; b != -1; b--) {
        result[j] = ((input >> b) & 1);
        j++;
#endif
    }
}

inline void p_btbe_128bit(XXH128_hash_t input_set, int* result) {
#if defined(IS_BIG_ENDIAN)
    p_btbe_64bit(input_set.low64, result);
    p_btbe_64bit(input_set.high64, result + 64);
#else
    p_btbe_64bit(input_set.high64, result);
    p_btbe_64bit(input_set.low64, result + 64);
#endif
}

inline void p_btbe_128bit(XXH128_hash_t input_set, const int output_size, int* result) {
#if defined(IS_BIG_ENDIAN)
    p_btbe<uint64_t>(input_set.low64, output_size, result);
    p_btbe<uint64_t>(input_set.high64, output_size, result + output_size);
#else
    p_btbe<uint64_t>(input_set.high64, output_size, result + output_size);
    p_btbe<uint64_t>(input_set.low64, output_size, result);
#endif
}


/*
###################################################################
#                           PUBLIC API                            #
###################################################################
*/

void str_to_binary_array(const std::string input, const int size, int* result) {
    int val = 0;
    size_t offset = 0;
    for (int i = 0; i < size; i++) {
        val = input[i];
        offset = i * 8;
#if defined(IS_BIG_ENDIAN)
        // first bit of variable is store at first location in memory
        for (int b = 0; b < 8; b++) {
            result[offset + b] = ((val >> b) & 1);
#else
        // first bit of variable is store at last location in memory
        int j = 0;
        for (int b = 7; b != -1; b--) {
            result[offset + j] = ((val >> b) & 1);
            j++;
#endif
        }
    }
}

/*
*******************************************************************
*                              64 bit                             *
*******************************************************************
*/

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_64bits(
    const std::string input, const int output_size, int* result, uint64_t seed = 0
) {
    // compute hash
    uint64_t hash = XXH3_64bits_withSeed(input.c_str(), input.size(), seed);
    p_btbe<uint64_t>(hash, output_size, result);
}

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_64bits(
    void* input, const int input_size, const int output_size, int* result, uint64_t seed = 0
) {
    // compute hash
    uint64_t hash = XXH3_64bits_withSeed(input, input_size, seed);
    p_btbe<uint64_t>(hash, output_size, result);
}

inline void hash_to_binary_array_64bits(
    void* input, const int input_size, int* result, uint64_t seed = 0
) {
    // compute hash
    uint64_t hash = XXH3_64bits_withSeed(input, input_size, seed);
    p_btbe_64bit(hash, result);
}

/*
*******************************************************************
*                             128 bit                             *
*******************************************************************
*/

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_128bits(
    const std::string input, const int output_size, int* result, uint64_t seed = 0
) {
    // compute hash
    XXH128_hash_t hash = XXH3_128bits_withSeed(input.c_str(), input.size(), seed);
    p_btbe_128bit(hash, output_size, result);
}

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_128bits(
    void* input, const int input_size, int* result, uint64_t seed = 0
) {
    // compute hash
    XXH128_hash_t hash = XXH3_128bits_withSeed(input, input_size, seed);
    p_btbe_128bit(hash, result);
}

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_128bits(
    void* input, const int input_size, const int output_size, int* result, uint64_t seed = 0
) {
    // compute hash
    XXH128_hash_t hash = XXH3_128bits_withSeed(input, input_size, seed);
    p_btbe_128bit(hash, output_size, result);
}
#endif  // INCLUDE_BEHCC_HPP_
