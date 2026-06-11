cmake_minimum_required (VERSION 3.20.1)

project(test-simulator-mhd)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  file(COPY "harris_2d_amr" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)
  phare_mpi_python3_exec(9 1 mhd-harris-2d-amr harris_2d_amr/case.py ${CMAKE_CURRENT_BINARY_DIR})

  if(testMPI)
    foreach(N 2 4)
      phare_mpi_python3_exec(9 ${N} mhd-harris-2d-amr harris_2d_amr/case.py ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
  endif()
endif()
