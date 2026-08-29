# statarb — C++17 port of the statistical-arbitrage toolkit.
# Targets: statarb (static lib), statarb_demo, statarb_tests (GoogleTest).
# Tests run with the working directory set to this source dir so the
# golden file resolves at ../data/golden/golden.json (per the contract).
cmake_minimum_required(VERSION 3.16)
project(statarb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

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

enable_testing()
find_package(GTest REQUIRED)

add_executable(statarb_tests
  tests/test_golden.cpp
  tests/test_ols.cpp
  tests/test_adf.cpp
  tests/test_signals.cpp
  tests/test_johansen.cpp
  tests/test_kalman.cpp
  tests/test_backtest.cpp)
target_include_directories(statarb_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(statarb_tests PRIVATE statarb GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(statarb_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
