cmake_minimum_required (VERSION 3.20.1)

project(test-tdtagged)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)

  ## These test use dump diagnostics so require HighFive!
  phare_python3_exec(11 test-td-tagged td1dtagged.py ${CMAKE_CURRENT_BINARY_DIR})

endif()
