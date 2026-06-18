cmake_minimum_required(VERSION 3.20.1)

set(HALL_COMMON_INCLUDES
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${SAMRAI_INCLUDE_DIRS}>
  ${GTEST_INCLUDE_DIRS}
)

set(HALL_COMMON_LIBS
  phare_amr
  phare_initializer
  phare_core
  ${SAMRAI_LIBRARIES}
  ${GTEST_LIBS}
)

add_executable(test-hall-unit-convergence test_hall_convergence.cpp)
target_include_directories(test-hall-unit-convergence PRIVATE ${HALL_COMMON_INCLUDES})
target_link_libraries(test-hall-unit-convergence PRIVATE ${HALL_COMMON_LIBS})
add_no_mpi_phare_test(test-hall-unit-convergence ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test-hall-compute-flux-convergence test_hall_compute_flux_convergence.cpp)
target_include_directories(test-hall-compute-flux-convergence PRIVATE ${HALL_COMMON_INCLUDES})
target_link_libraries(test-hall-compute-flux-convergence PRIVATE ${HALL_COMMON_LIBS})
add_no_mpi_phare_test(test-hall-compute-flux-convergence ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test-ideal-compute-flux-convergence test_ideal_compute_flux_convergence.cpp)
target_include_directories(test-ideal-compute-flux-convergence PRIVATE ${HALL_COMMON_INCLUDES})
target_link_libraries(test-ideal-compute-flux-convergence PRIVATE ${HALL_COMMON_LIBS})
add_no_mpi_phare_test(test-ideal-compute-flux-convergence ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test-instrumented-flux-pipeline test_instrumented_flux_pipeline.cpp)
target_include_directories(test-instrumented-flux-pipeline PRIVATE ${HALL_COMMON_INCLUDES})
target_link_libraries(test-instrumented-flux-pipeline PRIVATE ${HALL_COMMON_LIBS})
add_no_mpi_phare_test(test-instrumented-flux-pipeline ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test-decomposed-flux-pipeline test_decomposed_flux_pipeline.cpp)
target_include_directories(test-decomposed-flux-pipeline PRIVATE ${HALL_COMMON_INCLUDES})
target_link_libraries(test-decomposed-flux-pipeline PRIVATE ${HALL_COMMON_LIBS})
add_no_mpi_phare_test(test-decomposed-flux-pipeline ${CMAKE_CURRENT_BINARY_DIR})
