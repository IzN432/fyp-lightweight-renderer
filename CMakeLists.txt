cmake_minimum_required(VERSION 3.29.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(FypLightweightRenderer LANGUAGES CXX C)

find_package(Vulkan REQUIRED)
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

function(add_shaders TARGET)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/shaders")
    set(SPV_FILES)
    foreach(GLSL ${ARGN})
        get_filename_component(FNAME ${GLSL} NAME)
        set(SPV "${CMAKE_BINARY_DIR}/shaders/${FNAME}.spv")
        add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${GLSLC} ${CMAKE_CURRENT_SOURCE_DIR}/${GLSL} -o ${SPV}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${GLSL}
            COMMENT "Compiling shader ${GLSL}"
        )
        list(APPEND SPV_FILES ${SPV})
    endforeach()
    add_custom_target(${TARGET}_shaders DEPENDS ${SPV_FILES})
    add_dependencies(${TARGET} ${TARGET}_shaders)
endfunction()
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

# MSVC flags
add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>)
add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>)
add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:preprocessor>)

# Vendored libs (imgui, ImGuiFileDialog)
add_subdirectory(external)

# Core library
add_library(lr_core STATIC
    src/core/vulkan/VulkanContext.cpp
    src/core/vulkan/ShaderLoader.cpp
    src/core/vulkan/Allocator.cpp
    src/core/window/Window.cpp
)
target_include_directories(lr_core PUBLIC src)
target_link_libraries(lr_core PUBLIC
    Vulkan::Vulkan
    glfw
    spdlog::spdlog
    GPUOpen::VulkanMemoryAllocator
)
target_compile_definitions(lr_core PUBLIC
    LR_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders/"
)

add_shaders(lr_core
    src/core/passes/test.vert
)

# Test executable
add_executable(renderer src/main.cpp)
target_link_libraries(renderer PRIVATE lr_core)
