# add_subdirectory(vulkan_helper)

find_package(Vulkan)

add_executable(app
        main.cpp
        swapchain.hpp
        swapchain.cpp

        wrappers/window.hpp
        wrappers/window.cpp
        wrappers/gpu_device.hpp
        wrappers/gpu_device.cpp

        utils/vma_source.cpp
        utils/utils.hpp
        )
target_link_libraries(app
        PRIVATE compiler_options
        #vulkan_helper
        Vulkan::Vulkan
        beyond::core

        third_party::vma
        #third_party::volk

        vk-bootstrap::vk-bootstrap

        CONAN_PKG::glfw
        CONAN_PKG::glm
        CONAN_PKG::stb
        CONAN_PKG::tinyobjloader
        )

add_dependencies(app vertShader)
add_dependencies(app fragShader)
add_dependencies(app assets)

add_clangformat(app)
