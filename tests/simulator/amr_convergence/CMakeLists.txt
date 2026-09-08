cmake_minimum_required (VERSION 3.20.1)

project(test-simulator-amr-convergence)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)
  ## AMR space-time convergence studies (2D MHD); dump + read back h5 diagnostics.
  ## One ctest entry per test method so `ctest -j` runs the independent sweeps
  ## concurrently; inside an entry the sweep is sequential (order/drift need
  ## all points together).
  if(PHARE_PYTEST_SIMULATORS GREATER_EQUAL 2)
    phare_mpi_python3_exec(21 ${PHARE_MPI_PROCS} amr-convergence-alfven2d-spatial-o2      test_alfven2d_amr.py   ${CMAKE_CURRENT_BINARY_DIR} AlfvenConvergenceTest.test_spatial_convergence_1_2)
    phare_mpi_python3_exec(21 ${PHARE_MPI_PROCS} amr-convergence-alfven2d-spatial-o4      test_alfven2d_amr.py   ${CMAKE_CURRENT_BINARY_DIR} AlfvenConvergenceTest.test_spatial_convergence_2_4)
    phare_mpi_python3_exec(21 ${PHARE_MPI_PROCS} amr-convergence-alfven2d-temporal        test_alfven2d_amr.py   ${CMAKE_CURRENT_BINARY_DIR} AlfvenConvergenceTest.test_temporal_sigma_sweep)
    phare_mpi_python3_exec(21 ${PHARE_MPI_PROCS} amr-convergence-whistler2d-spatial-o2     test_whistler2d_amr.py ${CMAKE_CURRENT_BINARY_DIR} WhistlerConvergenceTest.test_spatial_convergence_1_2)
    phare_mpi_python3_exec(21 ${PHARE_MPI_PROCS} amr-convergence-whistler2d-spatial-o4     test_whistler2d_amr.py ${CMAKE_CURRENT_BINARY_DIR} WhistlerConvergenceTest.test_spatial_convergence_2_4)
    phare_mpi_python3_exec(21 ${PHARE_MPI_PROCS} amr-convergence-whistler2d-temporal       test_whistler2d_amr.py ${CMAKE_CURRENT_BINARY_DIR} WhistlerConvergenceTest.test_temporal_sigma_sweep)
    phare_mpi_python3_exec(21 ${PHARE_MPI_PROCS} amr-convergence-alfven2d-spatial-o4-iso    test_alfven2d_iso_amr.py   ${CMAKE_CURRENT_BINARY_DIR} AlfvenIsoConvergenceTest.test_spatial_convergence_order4_isotropic)
    phare_mpi_python3_exec(21 ${PHARE_MPI_PROCS} amr-convergence-whistler2d-spatial-o4-iso  test_whistler2d_iso_amr.py ${CMAKE_CURRENT_BINARY_DIR} WhistlerIsoConvergenceTest.test_spatial_convergence_order4_isotropic)
    phare_mpi_python3_exec(21 ${PHARE_MPI_PROCS} amr-convergence-alfven2d-uniform-spatial-o4   uniform_spatial_check.py ${CMAKE_CURRENT_BINARY_DIR} AlfvenUniformSpatial.test_uniform_spatial)
    phare_mpi_python3_exec(21 ${PHARE_MPI_PROCS} amr-convergence-whistler2d-uniform-spatial-o4 uniform_spatial_check.py ${CMAKE_CURRENT_BINARY_DIR} WhistlerUniformSpatial.test_uniform_spatial)
  endif()
endif()
