add_library(vulkan_helper
        instance.hpp
        instance.cpp
        queue_indices.hpp
        queue_indices.cpp
        swapchain.hpp
        swapchain.cpp
        utils.hpp panic.cpp panic.hpp)
target_link_libraries(vulkan_helper
        PUBLIC
        third_party::vma
        third_party::volk
        PRIVATE
        compiler_options)
add_clangformat(vulkan_helper)

set(VULKAN_HELPER_ENABLE_VALIDATION_LAYER AUTO CACHE STRING "The policy of enabling
    vulkan validation layer or not. AUTO means follow the debug or release build setting.")
set_property(CACHE VULKAN_HELPER_ENABLE_VALIDATION_LAYER PROPERTY STRINGS AUTO ON OFF)

if (VULKAN_HELPER_ENABLE_VALIDATION_LAYER STREQUAL "AUTO")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(vulkan_helper PRIVATE VULKAN_HELPER_ENABLE_VALIDATION_LAYER)
    endif ()
elseif (VULKAN_HELPER_ENABLE_VALIDATION_LAYER STREQUAL "ON")
    target_compile_definitions(vulkan_helper PRIVATE VULKAN_HELPER_ENABLE_VALIDATION_LAYER)
endif ()