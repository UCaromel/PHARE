cmake_minimum_required (VERSION 3.20.1)

project(test-boundary_condition)

set(SOURCES test_main.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
  ${GTEST_INCLUDE_DIRS}
  )

target_link_libraries(${PROJECT_NAME} PRIVATE
  phare_core
  ${GTEST_LIBS})

add_no_mpi_phare_test(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR})


set(FIELD_BC_TEST test-field-boundary-conditions)

add_executable(${FIELD_BC_TEST} test_field_boundary_conditions.cpp)

target_include_directories(${FIELD_BC_TEST} PRIVATE
  ${GTEST_INCLUDE_DIRS}
  ${PHARE_PROJECT_DIR}
  )

target_link_libraries(${FIELD_BC_TEST} PRIVATE
  phare_core
  ${GTEST_LIBS})

add_no_mpi_phare_test(${FIELD_BC_TEST} ${CMAKE_CURRENT_BINARY_DIR})


