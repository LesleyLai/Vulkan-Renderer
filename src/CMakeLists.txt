add_subdirectory(vulkan_helper)

find_package(Vulkan)

add_executable(app
        main.cpp
        window.hpp
        window.cpp
        swapchain.hpp
        swapchain.cpp
        gpu_device.hpp
        gpu_device.cpp
        )
target_link_libraries(app
        PRIVATE compiler_options
        #vulkan_helper
        Vulkan::Vulkan
        beyond::core
        CONAN_PKG::glfw
        CONAN_PKG::glm
        CONAN_PKG::stb
        CONAN_PKG::tinyobjloader
        )

add_dependencies(app vertShader)
add_dependencies(app fragShader)
add_dependencies(app assets)

add_clangformat(app)
