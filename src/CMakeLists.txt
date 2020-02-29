add_subdirectory(vulkan_helper)

add_executable(app
        main.cpp
        window.hpp
        window.cpp)
target_link_libraries(app
        PRIVATE compiler_options
        vulkan_helper
        beyond::core
        CONAN_PKG::glfw
        )
add_clangformat(app)
