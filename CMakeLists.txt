cmake_minimum_required(VERSION 3.20)
project(PhaseSpaceONNX LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

set(_HDF5_HINTS)
if(WIN32 AND EXISTS "C:/msys64/ucrt64")
    list(APPEND _HDF5_HINTS "C:/msys64/ucrt64")
endif()
find_package(HDF5 REQUIRED COMPONENTS C HINTS ${_HDF5_HINTS})
if(TARGET HDF5::HDF5)
    set(_PHASESPACE_HDF5_LIBRARIES HDF5::HDF5)
elseif(TARGET hdf5-shared)
    set(_PHASESPACE_HDF5_LIBRARIES hdf5-shared)
elseif(TARGET hdf5-static)
    set(_PHASESPACE_HDF5_LIBRARIES hdf5-static)
elseif(HDF5_LIBRARIES)
    set(_PHASESPACE_HDF5_LIBRARIES ${HDF5_LIBRARIES})
else()
    message(FATAL_ERROR "HDF5 was found but did not provide a C library target")
endif()

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime C++ SDK root directory")
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime C++ SDK root directory" FORCE)
endif()
if(NOT ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.deps/onnxruntime")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include"
)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime C++ SDK was not found. Set ONNXRUNTIME_ROOT to its installation directory."
    )
endif()
get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)

add_executable(phasespace_infer_bench
    cpp/src/phasespace_driver.cpp
    cpp/src/main_benchmark.cpp
)

target_include_directories(phasespace_infer_bench PRIVATE
    cpp/include
    "${ONNXRUNTIME_INCLUDE_DIR}"
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(phasespace_infer_bench PRIVATE
    OpenMP::OpenMP_CXX
    ${_PHASESPACE_HDF5_LIBRARIES}
    "${ONNXRUNTIME_LIBRARY}"
)

if(MSVC)
    target_compile_options(phasespace_infer_bench PRIVATE /W4 /arch:AVX2 $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(phasespace_infer_bench PRIVATE
        -Wall
        -Wextra
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
    )
    target_compile_definitions(phasespace_infer_bench PRIVATE $<$<CONFIG:Release>:NDEBUG>)
    if(WIN32)
        target_compile_definitions(phasespace_infer_bench PRIVATE _stdcall=)
    endif()
endif()

if(UNIX AND NOT APPLE)
    set_target_properties(phasespace_infer_bench PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
        INSTALL_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
    )
endif()

if(WIN32)
    find_file(ONNXRUNTIME_DLL
        NAMES onnxruntime.dll
        HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/bin"
    )
    if(ONNXRUNTIME_DLL)
        add_custom_command(TARGET phasespace_infer_bench POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_DLL}"
                "$<TARGET_FILE_DIR:phasespace_infer_bench>/onnxruntime.dll"
            COMMENT "Copying ONNX Runtime DLL beside the benchmark executable"
        )
    endif()
endif()
