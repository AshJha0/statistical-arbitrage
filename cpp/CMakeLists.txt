# statarb — C++17 port of the statistical-arbitrage toolkit.
# Targets: statarb (static lib), statarb_demo, statarb_tests (GoogleTest).
# Tests run with the working directory set to this source dir so the
# golden file resolves at ../data/golden/golden.json (per the contract).
cmake_minimum_required(VERSION 3.16)
project(statarb LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(STATARB_BUILD_TESTS "Build the GoogleTest suite" ON)
option(STATARB_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Bit-for-bit cross-language agreement: never fuse a*b+c into an FMA.
add_compile_options(-Wall -Wextra -ffp-contract=off)

if(STATARB_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(statarb
  src/linalg.cpp
  src/ols.cpp
  src/adf.cpp
  src/signals.cpp
  src/engle_granger.cpp
  src/johansen.cpp
  src/kalman.cpp
  src/backtest.cpp
  src/data_io.cpp)
target_include_directories(statarb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(statarb_demo src/demo.cpp)
target_link_libraries(statarb_demo PRIVATE statarb)

if(STATARB_BUILD_TESTS)
enable_testing()
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  message(STATUS "System GoogleTest not found; fetching v1.14.0")
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(statarb_tests
  tests/test_golden.cpp
  tests/test_ols.cpp
  tests/test_adf.cpp
  tests/test_signals.cpp
  tests/test_johansen.cpp
  tests/test_kalman.cpp
  tests/test_backtest.cpp
  tests/test_data_io.cpp
  tests/test_v12.cpp)
target_include_directories(statarb_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(statarb_tests PRIVATE statarb GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(statarb_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
