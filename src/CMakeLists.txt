#add_library(common
#    "${CMAKE_SOURCE_DIR}/include/dummy.hpp"
#    dummy.cpp)
#target_link_libraries(common PRIVATE compiler_options)
#target_include_directories(common PUBLIC "${CMAKE_SOURCE_DIR}/include")
#add_clangformat(common)

add_executable(app
        main.cpp
        vulkan_utils.hpp
        vulkan_queue_indices.hpp
        vulkan_queue_indices.cpp
        vulkan_swapchain.hpp
        vulkan_swapchain.cpp
        window.hpp
        window.cpp
        )
target_link_libraries(app
        PRIVATE compiler_options
        beyond::core
        third_party::vma
        third_party::volk
        CONAN_PKG::glfw
        )
add_clangformat(app)


set(VULKAN_RENDERER_ENABLE_VALIDATION_LAYER AUTO CACHE STRING "The policy of enabling
    vulkan validation layer or not. AUTO means follow the debug or release build setting.")
set_property(CACHE VULKAN_RENDERER_ENABLE_VALIDATION_LAYER PROPERTY STRINGS AUTO ON OFF)

if (VULKAN_RENDERER_ENABLE_VALIDATION_LAYER STREQUAL "AUTO")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(app PRIVATE VULKAN_RENDERER_ENABLE_VALIDATION_LAYER)
    endif ()
elseif (VULKAN_RENDERER_ENABLE_VALIDATION_LAYER STREQUAL "ON")
    target_compile_definitions(app PRIVATE VULKAN_RENDERER_ENABLE_VALIDATION_LAYER)
endif ()