add_library(vulkan_helper
        check.hpp
        gpu_device.hpp
        gpu_device.cpp
        shader_module.hpp
        shader_module.cpp
        swapchain.hpp
        swapchain.cpp
        unique_resource.hpp
        buffer.cpp buffer.hpp
        single_time_command.hpp single_time_command.cpp
        image.cpp image.hpp image_view.cpp image_view.hpp)
target_link_libraries(vulkan_helper
        PUBLIC
        beyond::core
        third_party::vma
        Vulkan::Vulkan
        vk-bootstrap::vk-bootstrap
        CONAN_PKG::assimp
        CONAN_PKG::fmt
        PRIVATE
        compiler_options)
add_clangformat(vulkan_helper)

set(VULKAN_HELPER_ENABLE_VALIDATION_LAYER AUTO CACHE STRING "The policy of enabling
    vulkan validation layer or not. AUTO means follow the debug or release build setting.")
set_property(CACHE VULKAN_HELPER_ENABLE_VALIDATION_LAYER PROPERTY STRINGS AUTO ON OFF)

if (VULKAN_HELPER_ENABLE_VALIDATION_LAYER STREQUAL "AUTO")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(vulkan_helper PUBLIC VULKAN_HELPER_ENABLE_VALIDATION_LAYER)
    endif ()
elseif (VULKAN_HELPER_ENABLE_VALIDATION_LAYER STREQUAL "ON")
    target_compile_definitions(vulkan_helper PUBLIC VULKAN_HELPER_ENABLE_VALIDATION_LAYER)
endif ()