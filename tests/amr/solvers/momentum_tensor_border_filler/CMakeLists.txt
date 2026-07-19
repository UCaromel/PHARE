cmake_minimum_required (VERSION 3.20.1)

project(test-momentum-tensor-border-filler)

configure_file(input_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/input_config.h)

set(SOURCES_CPP
  test_momentum_tensor_border_filler.cpp
   )

add_executable(${PROJECT_NAME} ${SOURCES_CPP})

target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries(${PROJECT_NAME} PRIVATE
  phare_amr
  phare_initializer
  ${GTEST_LIBS})

add_phare_test(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR})
