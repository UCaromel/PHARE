cmake_minimum_required(VERSION 3.20.1)
project(test-simulator-reflux)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)
  phare_mpi_python3_exec(9 ${PHARE_MPI_PROCS} reflux-1d test_reflux_1d.py ${CMAKE_CURRENT_BINARY_DIR})
  phare_mpi_python3_exec(9 ${PHARE_MPI_PROCS} reflux-2d test_reflux_2d.py ${CMAKE_CURRENT_BINARY_DIR})
  phare_mpi_python3_exec(9 ${PHARE_MPI_PROCS} reflux-3d test_reflux_3d.py ${CMAKE_CURRENT_BINARY_DIR})
endif()
