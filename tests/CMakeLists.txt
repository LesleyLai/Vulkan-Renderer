add_library(vkh_test
        main.cpp)

target_link_libraries(vkh_test
        vulkan_helper
        CONAN_PKG::Catch2)