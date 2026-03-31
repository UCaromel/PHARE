cmake_minimum_required (VERSION 3.20.1)

project(test-mhd_ssi)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)
  phare_python3_exec(111, test-mhd_ssi_shock      ssi_shock.py      ${CMAKE_CURRENT_BINARY_DIR})
  phare_python3_exec(111, test-mhd_ssi_convergence ssi_convergence.py ${CMAKE_CURRENT_BINARY_DIR})
endif()
