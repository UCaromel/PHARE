cmake_minimum_required (VERSION 3.20.1)
project(phare_python3)

if(NOT PHARE_PERMUTATIONS)
  set(PHARE_PERMUTATIONS "${PHARE_PROJECT_DIR}/res/sim/all.txt")
endif()

file(STRINGS ${PHARE_PERMUTATIONS} THIS_FILE)

while(THIS_FILE)
    list(POP_FRONT THIS_FILE LINE)

    # Skip empty lines and comments
    string(STRIP "${LINE}" LINE)
    if("${LINE}" STREQUAL "" OR "${LINE}" MATCHES "^#")
        continue()
    endif()

    string(REPLACE "," ";" LINE_LIST ${LINE})
    list(LENGTH LINE_LIST LINE_LENGTH)

    # Auto-detect model type by line length:
    # Length == 3: Hybrid-only (dim,interp,nbpart)
    # Length == 8: MHD-only (dim,timeint,recon,slope,riemann,hall,res,hyper)
    # Length == 10: Both models (dim,interp,nbpart,timeint,recon,slope,riemann,hall,res,hyper)
    
    if(LINE_LENGTH EQUAL 3)
        # Hybrid-only: fill in MHD defaults
        list(APPEND LINE_LIST "TimeIntegratorType::Default")
        list(APPEND LINE_LIST "ReconstructionType::Default")
        list(APPEND LINE_LIST "SlopeLimiterType::None")
        list(APPEND LINE_LIST "RiemannSolverType::Default")
        list(APPEND LINE_LIST "false")  # Hall
        list(APPEND LINE_LIST "false")  # Resistivity
        list(APPEND LINE_LIST "false")  # HyperResistivity
    elseif(LINE_LENGTH EQUAL 8)
        # MHD-only: prepend dummy interp=1, nbpart=2 for template compatibility
        # The actual MHD code path won't use these values
        list(GET LINE_LIST 0 DIM)
        list(REMOVE_AT LINE_LIST 0)
        list(PREPEND LINE_LIST ${DIM} "1" "2")
        # Add enum prefixes to MHD params (now at positions 3-6)
        SET(_OFFSET 3)
        SET(_PREFIXES "TimeIntegratorType;ReconstructionType;SlopeLimiterType;RiemannSolverType")
        foreach(PREFIX IN LISTS _PREFIXES)
            list(TRANSFORM LINE_LIST PREPEND "${PREFIX}::" AT ${_OFFSET})
            MATH(EXPR _OFFSET "${_OFFSET}+1")
        endforeach()
    elseif(LINE_LENGTH EQUAL 10)
        # Both models: add enum prefixes (current format)
        SET(_OFFSET 3)
        SET(_PREFIXES "TimeIntegratorType;ReconstructionType;SlopeLimiterType;RiemannSolverType")
        foreach(PREFIX IN LISTS _PREFIXES)
            list(TRANSFORM LINE_LIST PREPEND "${PREFIX}::" AT ${_OFFSET})
            MATH(EXPR _OFFSET "${_OFFSET}+1")
        endforeach()
    else()
        message(FATAL_ERROR "Invalid permutation line: ${LINE}. Expected 3 (Hybrid), 8 (MHD), or 10 (Both) values.")
    endif()

    string(REPLACE "," "_" OUT ${LINE})
    SET(MOD_STR "cpp_${OUT}")
    list(JOIN LINE_LIST "," LINE) # semicolon separated list to csv

    pybind11_add_module(${MOD_STR} cpp_simulator.cpp)
    target_link_libraries(${MOD_STR} PUBLIC phare_simulator)
    target_compile_options(${MOD_STR} PRIVATE
        ${PHARE_WERROR_FLAGS}
        -DPHARE_HAS_HIGHFIVE=${PHARE_HAS_HIGHFIVE}
        -DPHARE_SIM_ID=${OUT}
        "-DPHARE_SIM_STR=${LINE}"
    )
    set_target_properties(${MOD_STR}
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pybindlibs"
    )
    # this is on by default "pybind11_add_module" but can interfere with coverage so we disable it if coverage is enabled
    set_property(TARGET ${MOD_STR} PROPERTY INTERPROCEDURAL_OPTIMIZATION ${PHARE_INTERPROCEDURAL_OPTIMIZATION})
    set_property(TARGET ${MOD_STR} APPEND_STRING PROPERTY LINK_FLAGS " ${PHARE_LINK_FLAGS}")
endwhile()

# cpp_etc == everything not simulator related!
pybind11_add_module(cpp_etc cpp_etc.cpp)
target_compile_options(cpp_etc PRIVATE ${PHARE_WERROR_FLAGS} -DPHARE_HAS_HIGHFIVE=${PHARE_HAS_HIGHFIVE})
target_link_libraries(cpp_etc PUBLIC phare_amr phare_diagnostic)
set_target_properties(cpp_etc
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pybindlibs"
)
# this is on by default "pybind11_add_module" but can interfere with coverage so we disable it if coverage is enabled
set_property(TARGET cpp_etc PROPERTY INTERPROCEDURAL_OPTIMIZATION ${PHARE_INTERPROCEDURAL_OPTIMIZATION})
set_property(TARGET cpp_etc APPEND_STRING PROPERTY LINK_FLAGS " ${PHARE_LINK_FLAGS}")
