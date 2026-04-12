cmake_minimum_required (VERSION 3.20.1)

project(test-mhd-hybrid-harris)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive AND testMPI)
  if(testMPI)
    phare_mpi_python3_exec(11 1 mhd_hybrid_harris_2d mhd_hybrid_harris_2d.py ${CMAKE_CURRENT_BINARY_DIR})
  endif(testMPI)
endif()
