cmake_minimum_required (VERSION 3.20.1)

project(test-amr-solvers)

add_subdirectory(convergence)

add_executable(test-mhd-profile-resources test_mhd_profile_resources.cpp)
target_include_directories(test-mhd-profile-resources PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(test-mhd-profile-resources PRIVATE
  phare_simulator
  pybind11::embed
  ${GTEST_LIBS}
)
add_phare_test(test-mhd-profile-resources ${CMAKE_CURRENT_BINARY_DIR})
