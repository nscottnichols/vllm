if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR must be defined")
endif()

set(CMAKE_LISTS "${SOURCE_DIR}/CMakeLists.txt")
if(NOT EXISTS "${CMAKE_LISTS}")
    message(FATAL_ERROR "Missing vLLM flash-attention CMakeLists: ${CMAKE_LISTS}")
endif()

file(READ "${CMAKE_LISTS}" CMAKE_LISTS_CONTENT)
if(CMAKE_LISTS_CONTENT MATCHES "VLLM_FA2_SM12_FAMILY_PATCH")
    return()
endif()

set(ORIGINAL
    "cuda_archs_loose_intersection(FA2_ARCHS \"8.0+PTX\" \"\${CUDA_ARCHS}\")"
)
set(REPLACEMENT
    "if(\"\${CUDA_ARCHS}\" MATCHES \"^12\\\\.\")
        set(FA2_ARCHS \"12.0f\")  # VLLM_FA2_SM12_FAMILY_PATCH
        set(FA3_ENABLED OFF)
    else()
        cuda_archs_loose_intersection(FA2_ARCHS \"8.0+PTX\" \"\${CUDA_ARCHS}\")
    endif()"
)

string(REPLACE "${ORIGINAL}" "${REPLACEMENT}" PATCHED_CONTENT "${CMAKE_LISTS_CONTENT}")
if(PATCHED_CONTENT STREQUAL CMAKE_LISTS_CONTENT)
    message(FATAL_ERROR "Could not patch FA2 architecture selection")
endif()

file(WRITE "${CMAKE_LISTS}" "${PATCHED_CONTENT}")
