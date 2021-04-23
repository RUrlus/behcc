/*
    include/behcc/behcc.h: Binary Encoding for High Cardinality Categoricals

    Copyright (c) 2021 Ralph Urlus <rurlus.dev@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

*/
#ifndef INCLUDE_BEHCC_BEHCC_H_
#define INCLUDE_BEHCC_BEHCC_H_

#include <xxhash.h>

#include <cmath>
#include <string>
#include <algorithm>
#include <type_traits>

namespace behcc {
/*
###################################################################
#                    Private utility functions                    #
###################################################################
*/
namespace pr {

/* */
template <typename T>
inline void btbe(T input, const int nbits, int* result) {
#if defined(IS_BIG_ENDIAN)
    // first bit of variable is store at first location in memory
    for (int b = 0; b < nbits; b++) {
        result[b] = ((input >> b) & 1);
#else
    // first bit of variable is store at last location in memory
    int j = 0;
    for (int b = nbits; b-- > 0; ) {
        result[j] = ((input >> b) & 1);
        j++;
#endif
    }
}

template <typename T>
inline void btbe_8bit(T input, int* result) {
#if defined(IS_BIG_ENDIAN)
    // first bit of variable is stored at first location in memory
    result[0] = ((input >> 0) & 1);
    result[1] = ((input >> 1) & 1);
    result[2] = ((input >> 2) & 1);
    result[3] = ((input >> 3) & 1);
    result[4] = ((input >> 4) & 1);
    result[5] = ((input >> 5) & 1);
    result[6] = ((input >> 6) & 1);
    result[7] = ((input >> 7) & 1);
#else
    // first bit of variable is stored at last location in memory
    result[0] = ((input >> 7) & 1);
    result[1] = ((input >> 6) & 1);
    result[2] = ((input >> 5) & 1);
    result[3] = ((input >> 4) & 1);
    result[4] = ((input >> 3) & 1);
    result[5] = ((input >> 2) & 1);
    result[6] = ((input >> 1) & 1);
    result[7] = ((input >> 0) & 1);
#endif
}

template <typename T, typename = typename std::enable_if< sizeof(T) >= 2 >::type>
inline void btbe_16bit(T input, int* result) {
#if defined(IS_BIG_ENDIAN)
    // first bit of variable is stored at first location in memory
    result[0] = ((input >> 0) & 1);
    result[1] = ((input >> 1) & 1);
    result[2] = ((input >> 2) & 1);
    result[3] = ((input >> 3) & 1);
    result[4] = ((input >> 4) & 1);
    result[5] = ((input >> 5) & 1);
    result[6] = ((input >> 6) & 1);
    result[7] = ((input >> 7) & 1);
    result[8] = ((input >> 8) & 1);
    result[9] = ((input >> 9) & 1);
    result[10] = ((input >> 10) & 1);
    result[11] = ((input >> 11) & 1);
    result[12] = ((input >> 12) & 1);
    result[13] = ((input >> 13) & 1);
    result[14] = ((input >> 14) & 1);
    result[15] = ((input >> 15) & 1);
#else
    result[0] = ((input >> 15) & 1);
    result[1] = ((input >> 14) & 1);
    result[2] = ((input >> 13) & 1);
    result[3] = ((input >> 12) & 1);
    result[4] = ((input >> 11) & 1);
    result[5] = ((input >> 10) & 1);
    result[6] = ((input >> 9) & 1);
    result[7] = ((input >> 8) & 1);
    result[8] = ((input >> 7) & 1);
    result[9] = ((input >> 6) & 1);
    result[10] = ((input >> 5) & 1);
    result[11] = ((input >> 4) & 1);
    result[12] = ((input >> 3) & 1);
    result[13] = ((input >> 2) & 1);
    result[14] = ((input >> 1) & 1);
    result[15] = ((input >> 0) & 1);
#endif
}

inline void btbe_32bit(uint64_t input, int* result) {
#if defined(is_big_endian)
    // first bit of variable is store at first location in memory
    for (int b = 0; b < 32; b++) {
        result[b] = ((input >> b) & 1);
#else
    // first bit of variable is store at last location in memory
    int j = 0;

    for (int b = 32; b-- > 0; ) {
        result[j] = ((input >> b) & 1);
        j++;
#endif
    }
}

template <typename T, typename std::enable_if< sizeof(T) >= 4 >::type>
inline void btbe_32bit(T input, int* result) {
#if defined(is_big_endian)
    // first bit of variable is store at first location in memory
    for (int b = 0; b < 32; b++) {
        result[b] = ((input >> b) & 1);
#else
    // first bit of variable is store at last location in memory
    int j = 0;
    for (int b = 32; b-- > 0; ) {
        result[j] = ((input >> b) & 1);
        j++;
#endif
    }
}

inline void btbe_64bit(uint64_t input, int* result) {
#if defined(is_big_endian)
    // first bit of variable is store at first location in memory
    for (int b = 0; b < 64; b++) {
        result[b] = ((input >> b) & 1);
#else
    // first bit of variable is store at last location in memory
    int j = 0;
    for (int b = 64; b-- > 0; ) {
        result[j] = ((input >> b) & 1);
        j++;
#endif
    }
}

template <typename T, typename = typename std::enable_if< sizeof(T) >= 8 >::type>
inline void btbe_64bit(T input, int* result) {
#if defined(IS_BIG_ENDIAN)
    // first bit of variable is store at first location in memory
    for (int b = 0; b < 64; b++) {
        result[b] = ((input >> b) & 1);
#else
    // first bit of variable is store at last location in memory
    int j = 0;
    for (int b = 64; b-- > 0; ) {
        result[j] = ((input >> b) & 1);
        j++;
#endif
    }
}

inline void btbe_128bit(XXH128_hash_t input_set, int* result) {
#if defined(IS_BIG_ENDIAN)
    btbe_64bit(input_set.low64, result);
    btbe_64bit(input_set.high64, result + 64);
#else
    btbe_64bit<uint64_t>(input_set.high64, result);
    btbe_64bit<uint64_t>(input_set.low64, result + 64);
#endif
}

inline void btbe_128bit(XXH128_hash_t input_set, const int output_size, int* result) {
#if defined(IS_BIG_ENDIAN)
    btbe<uint64_t>(input_set.low64, output_size, result);
    btbe<uint64_t>(input_set.high64, output_size, result + output_size);
#else
    btbe<uint64_t>(input_set.high64, output_size, result + output_size);
    btbe<uint64_t>(input_set.low64, output_size, result);
#endif
}

}  // namespace pr


/*
###################################################################
#                           PUBLIC API                            #
###################################################################
*/

inline void str_to_binary_array(const std::string input, const int size, int* result) {
    for (int i = 0; i < size; i++) {
        pr::btbe_8bit<int>(input[i], result);
        result += 8;
    }
}

/*
*******************************************************************
*                               8 bit                             *
*******************************************************************
*/

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_8bits(
    const std::string input, int* result, uint64_t seed = 0
) {
    // compute hash
    uint64_t hash = XXH3_64bits_withSeed(input.c_str(), input.size(), seed);
    pr::btbe_8bit<uint64_t>(hash, result);
}

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_8bits(
    void* input, const int input_size, int* result, uint64_t seed = 0
) {
    // compute hash
    uint64_t hash = XXH3_64bits_withSeed(input, input_size, seed);
    pr::btbe_8bit<uint64_t>(hash, result);
}

/*
*******************************************************************
*                              16 bit                             *
*******************************************************************
*/

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_16bits(
    const std::string input, int* result, uint64_t seed = 0
) {
    // compute hash
    uint64_t hash = XXH3_64bits_withSeed(input.c_str(), input.size(), seed);
    pr::btbe_16bit<uint64_t>(hash, result);
}

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_16bits(
    void* input, const int input_size, int* result, uint64_t seed = 0
) {
    // compute hash
    uint64_t hash = XXH3_64bits_withSeed(input, input_size, seed);
    pr::btbe_16bit<uint64_t>(hash, result);
}

/*
*******************************************************************
*                              32 bit                             *
*******************************************************************
*/

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_32bits(
    const std::string input, int* result, uint64_t seed = 0
) {
    // compute hash
    uint64_t hash = XXH3_64bits_withSeed(input.c_str(), input.size(), seed);
    pr::btbe_32bit(hash, result);
}

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_32bits(
    void* input, const int input_size, int* result, uint64_t seed = 0
) {
    // compute hash
    uint64_t hash = XXH3_64bits_withSeed(input, input_size, seed);
    pr::btbe_32bit(hash, result);
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
    pr::btbe<uint64_t>(hash, output_size, result);
}

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_64bits(
    void* input, const int input_size, const int output_size, int* result, uint64_t seed = 0
) {
    // compute hash
    uint64_t hash = XXH3_64bits_withSeed(input, input_size, seed);
    pr::btbe<uint64_t>(hash, output_size, result);
}

inline void hash_to_binary_array_64bits(
    void* input, const int input_size, int* result, uint64_t seed = 0
) {
    // compute hash
    uint64_t hash = XXH3_64bits_withSeed(input, input_size, seed);
    pr::btbe_64bit<uint64_t>(hash, result);
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
    pr::btbe_128bit(hash, output_size, result);
}

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_128bits(
    void* input, const int input_size, int* result, uint64_t seed = 0
) {
    // compute hash
    XXH128_hash_t hash = XXH3_128bits_withSeed(input, input_size, seed);
    pr::btbe_128bit(hash, result);
}

/* Store bit encoding of hash of input as integers in result */
inline void hash_to_binary_array_128bits(
    void* input, const int input_size, const int output_size, int* result, uint64_t seed = 0
) {
    // compute hash
    XXH128_hash_t hash = XXH3_128bits_withSeed(input, input_size, seed);
    pr::btbe_128bit(hash, output_size, result);
}

}  // namespace behcc
#endif  // INCLUDE_BEHCC_BEHCC_H_
