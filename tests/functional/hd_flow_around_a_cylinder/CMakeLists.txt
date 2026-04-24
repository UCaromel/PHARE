cmake_minimum_required (VERSION 3.20.1)

project(test-hd_flow_around_a_cylinder)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)

  ## 2D supersonic HD flow around a cylinder (inner boundary + shocks).
  phare_python3_exec(111, test-hd_flow_around_a_cylinder case.py ${CMAKE_CURRENT_BINARY_DIR})

endif()
