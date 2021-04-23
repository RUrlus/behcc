CMAKE_MINIMUM_REQUIRED(VERSION 3.4...3.18)

# ##############################################################################
#                                  EXECUTABLE                                  #
# ##############################################################################
SET(MODNAME "behcc_extension")
ADD_DEFINITIONS(-DEXTENSION_MODULE_NAME=${MODNAME})
SET(PROJ_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)
SET(EXT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern)


pybind11_add_module(${MODNAME}
    MODULE
        ${PROJECT_SOURCE_DIR}/src/bindings.cpp
        ${PROJECT_SOURCE_DIR}/src/hash_encoder.cpp
)

# PYBIT_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
TARGET_COMPILE_DEFINITIONS(${MODNAME} PRIVATE VERSION_INFO=${BEHCC_VERSION_INFO})

TARGET_LINK_LIBRARIES(${MODNAME} PRIVATE
    pybind11::headers
    Python3::Module
    Python3::NumPy
    xxHash::xxhash
)

TARGET_INCLUDE_DIRECTORIES(${MODNAME} PRIVATE
    ${EXT_INCLUDE_DIR}/xxHash
)

TARGET_INCLUDE_DIRECTORIES(${MODNAME} PUBLIC
    # path of the headers after installation
    $<INSTALL_INTERFACE:include> # <prefix>/include/carma
    # path of the headers before installation
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)


# ##############################################################################
#                                   INSTALL                                    #
# ##############################################################################
INSTALL(TARGETS ${MODNAME} DESTINATION .)
