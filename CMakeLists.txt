cmake_minimum_required(VERSION 3.14)

project(vulkan_renderer VERSION 0.0.2 LANGUAGES CXX)

include("cmake/StandardProjectSettings.cmake")
include("CMakeDependentOption")

include("cmake/compiler.cmake")
include("cmake/clangformat.cmake")
include("cmake/StaticAnalyzers.cmake")

include("cmake/conan.cmake")
run_conan()

include(cmake/CompileShader.cmake)
compile_shader(vertShader
        SOURCE ${CMAKE_SOURCE_DIR}/shaders/shader.vert
        TARGET ${CMAKE_BINARY_DIR}/bin/shaders/shader.vert.spv
        )

compile_shader(fragShader
        SOURCE ${CMAKE_SOURCE_DIR}/shaders/shader.frag
        TARGET ${CMAKE_BINARY_DIR}/bin/shaders/shader.frag.spv
        )


# Copy assets
add_custom_target(assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data ${CMAKE_BINARY_DIR}/bin
        )


add_subdirectory(third-party)
add_subdirectory(src)


if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif ()


if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif ()