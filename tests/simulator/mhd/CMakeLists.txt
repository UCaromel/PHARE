cmake_minimum_required (VERSION 3.20.1)

project(test-simulator-mhd)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  file(COPY "shock_1d" "orszag_tang_2d" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)
  phare_mpi_python3_exec(9 1 mhd-shock-1d       shock_1d/case.py       ${CMAKE_CURRENT_BINARY_DIR})
  phare_mpi_python3_exec(9 1 mhd-orszag-tang-2d orszag_tang_2d/case.py ${CMAKE_CURRENT_BINARY_DIR})

  if(testMPI)
    phare_mpi_python3_exec(9 ${PHARE_MPI_PROCS} mhd-shock-1d       shock_1d/case.py       ${CMAKE_CURRENT_BINARY_DIR})
    phare_mpi_python3_exec(9 ${PHARE_MPI_PROCS} mhd-orszag-tang-2d orszag_tang_2d/case.py ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endif()
