cmake_minimum_required(VERSION 3.21)

# =============================================================================
# Project Information
# =============================================================================
project(HashEncoder
    VERSION 0.1.0
    DESCRIPTION "Multi-resolution hash grid encoding library (Instant-NGP style) built on EasyGPU"
    LANGUAGES CXX C
)

# =============================================================================
# Options
# =============================================================================
option(HASHENCODER_BUILD_EXAMPLES "Build example programs" ON)
option(HASHENCODER_BUILD_TESTS "Build test suite" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Compiler Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8)
    add_compile_options(/wd4819)
endif()

# =============================================================================
# EasyGPU Dependency
# =============================================================================
# Disable EasyGPU components we don't need
set(EASYGPU_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(EASYGPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EASYGPU_BUILD_WINDOW OFF CACHE BOOL "" FORCE)
set(EASYGPU_BUILD_WINDOW_EXAMPLES OFF CACHE BOOL "" FORCE)
set(EASYGPU_BUILD_FRAGMENT_TESTER OFF CACHE BOOL "" FORCE)

# Only add EasyGPU if the parent project hasn't already done so.
# This allows multiple EasyGPU-dependent libraries to coexist in one build.
if(NOT TARGET EasyGPU)
    add_subdirectory(EasyGPU)
endif()

# Ensure EasyGPU target has conformant preprocessor (some subdirectory
# builds need this repeated at the parent scope).
if(MSVC)
    target_compile_options(EasyGPU PUBLIC /Zc:preprocessor)
endif()

# =============================================================================
# HashEncoder Library
# =============================================================================
add_library(HashEncoder STATIC
    include/HashEncoder/HashEncoder.h
    include/HashEncoder/HashGridConfig.h
    include/HashEncoder/HashGridEncoding.h
    include/HashEncoder/HashGridEncoding.inl
    source/HashGridEncoding.cpp
)

target_compile_features(HashEncoder PUBLIC cxx_std_20)
target_link_libraries(HashEncoder PUBLIC EasyGPU)

target_include_directories(HashEncoder PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(HashEncoder PUBLIC /Zc:preprocessor)
endif()

# =============================================================================
# Tests
# =============================================================================
if(HASHENCODER_BUILD_TESTS)
    enable_testing()

    # GPU forward + backward correctness tests
    add_executable(TestMinimalGPU tests/TestMinimalGPU.cpp)
    target_link_libraries(TestMinimalGPU PRIVATE HashEncoder)
    if(MSVC)
        target_compile_options(TestMinimalGPU PRIVATE /Zc:preprocessor)
    endif()

    # Core CPU tests (validation, encode, backward, serialization, etc.)
    add_executable(TestHashGridEncoding tests/TestHashGridEncoding.cpp)
    target_link_libraries(TestHashGridEncoding PRIVATE HashEncoder)
    if(MSVC)
        target_compile_options(TestHashGridEncoding PRIVATE /Zc:preprocessor)
    endif()
    add_test(NAME TestHashGridEncoding COMMAND TestHashGridEncoding)

    # Stress / edge-case tests (CPU only)
    add_executable(TestHashGridStress tests/TestHashGridStress.cpp)
    target_link_libraries(TestHashGridStress PRIVATE HashEncoder)
    if(MSVC)
        target_compile_options(TestHashGridStress PRIVATE /Zc:preprocessor)
    endif()
    add_test(NAME TestHashGridStress COMMAND TestHashGridStress)
endif()

# =============================================================================
# Examples
# =============================================================================
if(HASHENCODER_BUILD_EXAMPLES)
    add_executable(hash_grid_demo examples/hash_grid_demo/main.cpp)
    target_link_libraries(hash_grid_demo PRIVATE HashEncoder)
    if(MSVC)
        target_compile_options(hash_grid_demo PRIVATE /Zc:preprocessor)
    endif()
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "HashEncoder Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Examples: ${HASHENCODER_BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${HASHENCODER_BUILD_TESTS}")
message(STATUS "")
