cmake_minimum_required (VERSION 3.20.1)

add_executable(test-field_neumann_inner_boundary_condition test_field_neumann_inner_boundary_condition.cpp)

target_include_directories(test-field_neumann_inner_boundary_condition PRIVATE
  ${GTEST_INCLUDE_DIRS}
  )

target_link_libraries(test-field_neumann_inner_boundary_condition PRIVATE
  phare_core
  phare_simulator
  ${GTEST_LIBS})

add_no_mpi_phare_test(test-field_neumann_inner_boundary_condition ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test-field_dirichlet_inner_boundary_condition test_field_dirichlet_inner_boundary_condition.cpp)

target_include_directories(test-field_dirichlet_inner_boundary_condition PRIVATE
  ${GTEST_INCLUDE_DIRS}
  )

target_link_libraries(test-field_dirichlet_inner_boundary_condition PRIVATE
  phare_core
  phare_simulator
  ${GTEST_LIBS})

add_no_mpi_phare_test(test-field_dirichlet_inner_boundary_condition ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test-field_symmetric_inner_boundary_condition test_field_symmetric_inner_boundary_condition.cpp)

target_include_directories(test-field_symmetric_inner_boundary_condition PRIVATE
  ${GTEST_INCLUDE_DIRS}
  )

target_link_libraries(test-field_symmetric_inner_boundary_condition PRIVATE
  phare_core
  phare_simulator
  ${GTEST_LIBS})

add_no_mpi_phare_test(test-field_symmetric_inner_boundary_condition ${CMAKE_CURRENT_BINARY_DIR})
