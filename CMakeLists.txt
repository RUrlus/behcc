CMAKE_MINIMUM_REQUIRED(VERSION 3.4...3.18)

PROJECT(
    behcc
    VERSION 0.1.0
    DESCRIPTION "Compute binary encoding for high cardinality categoricals"
    LANGUAGES CXX
)
SET(CMAKE_CXX_STANDARD 14)
# We've set a hint in the setup.py such that CMake knows where to find pybind11
FIND_PACKAGE(pybind11 CONFIG REQUIRED)

# Include some CMake module to make our life easier
LIST(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")
	
# Set a default build type if none was specified
SET(DEFAULT_BUILD_TYPE Release)
INCLUDE(BuildType)

IF(CMAKE_BUILD_TYPE MATCHES "Release")
    # Someone wrote so nice CMake code to set the appropriate flags 
    # Use with care, this includes some aggressive optimization flags
    INCLUDE(OptimizeForArchitecture)
    OptimizeForArchitecture()
ENDIF()

FIND_PACKAGE(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Build static lib of xxhash
SET(XXHASH_BUNDLED_MODE ON)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/extern/xxhash/cmake_unofficial)

# Needed to ensure the same Python executable is found by Pybind11
IF (NOT DEFINED PYTHON_EXECUTABLE)
    SET(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
ENDIF ()

# Trigger the build of the Python extension
INCLUDE(${PROJECT_SOURCE_DIR}/src/CMakeLists.txt)
