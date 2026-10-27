cmake_minimum_required (VERSION 3.20.1)

project(test-mhd-regrid-time)

set(SOURCES_CPP
  test_mhd_regrid_time.cpp
   )

add_executable(${PROJECT_NAME} ${SOURCES_CPP})

add_dependencies(${PROJECT_NAME} cpp_etc)

target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  ${GTEST_INCLUDE_DIRS}
  )

target_link_libraries(${PROJECT_NAME} PRIVATE
  phare_simulator
  pybind11::embed
  ${GTEST_LIBS}
)


add_no_mpi_phare_test(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/job.py ${CMAKE_CURRENT_BINARY_DIR}/job.py @ONLY)


# the same test at fourth order
set(O4_TARGET ${PROJECT_NAME}-o4)
add_executable(${O4_TARGET} ${SOURCES_CPP})
add_dependencies(${O4_TARGET} cpp_etc)
target_compile_definitions(${O4_TARGET} PRIVATE PHARE_TEST_MHD_ORDER=4)
target_include_directories(${O4_TARGET} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  ${GTEST_INCLUDE_DIRS}
  )
target_link_libraries(${O4_TARGET} PRIVATE
  phare_simulator
  pybind11::embed
  ${GTEST_LIBS}
)
add_no_mpi_phare_test(${O4_TARGET} ${CMAKE_CURRENT_BINARY_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/job_o4.py ${CMAKE_CURRENT_BINARY_DIR}/job_o4.py @ONLY)
