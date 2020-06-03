find_package(Vulkan)

add_subdirectory(vulkan_helper)

add_executable(app
        main.cpp
        window.hpp
        window.cpp
        utils/vma_source.cpp
        utils/utils.hpp
        mesh.cpp
        mesh.hpp
        model.cpp
        model.hpp)
target_link_libraries(app
        PRIVATE compiler_options
        vulkan_helper
        Vulkan::Vulkan
        beyond::core
        third_party::vma
        vk-bootstrap::vk-bootstrap

        CONAN_PKG::glfw
        CONAN_PKG::glm
        CONAN_PKG::stb
        )

add_dependencies(app vertShader)
add_dependencies(app fragShader)
add_dependencies(app assets)

add_clangformat(app)
