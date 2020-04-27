add_library(vulkan_helper
        check.hpp
        instance.hpp
        instance.cpp
        ../wrappers/shader_module.hpp
        ../wrappers/shader_module.cpp
        physical_device_picker.hpp
        physical_device_picker.cpp
        logical_device.hpp
        logical_device.cpp
        panic.cpp panic.hpp ../wrappers/unique_resource.hpp)
target_link_libraries(vulkan_helper
        PUBLIC
        third_party::vma
        third_party::volk
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