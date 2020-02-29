#add_library(common
#    "${CMAKE_SOURCE_DIR}/include/dummy.hpp"
#    dummy.cpp)
#target_link_libraries(common PRIVATE compiler_options)
#target_include_directories(common PUBLIC "${CMAKE_SOURCE_DIR}/include")
#add_clangformat(common)

add_executable(app "main.cpp")
target_link_libraries(app
    PRIVATE compiler_options)
add_clangformat(app)
