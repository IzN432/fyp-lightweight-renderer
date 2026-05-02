cmake_minimum_required(VERSION 3.29.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(FypLightweightRenderer LANGUAGES CXX C)

find_package(Vulkan REQUIRED)
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

function(add_shaders TARGET)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/shaders")
    
    # Collect all shader headers (.glslh files) as implicit dependencies
    file(GLOB_RECURSE SHADER_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/core/passes/*.glslh")
    
    set(SPV_FILES)
    foreach(GLSL ${ARGN})
        get_filename_component(FNAME ${GLSL} NAME)
        set(SPV "${CMAKE_BINARY_DIR}/shaders/${FNAME}.spv")
        add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${GLSLC} ${CMAKE_CURRENT_SOURCE_DIR}/${GLSL} -o ${SPV}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${GLSL} ${SHADER_HEADERS}
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
find_package(mikktspace CONFIG REQUIRED)
find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h" REQUIRED)

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
    src/core/vulkan/CommandBuffer.cpp
    src/core/vulkan/DescriptorAllocator.cpp
    src/core/vulkan/Swapchain.cpp
    src/core/vulkan/Renderer.cpp

    src/core/framegraph/ResourceRegistry.cpp
    src/core/framegraph/FrameGraph.cpp

    src/core/pipeline/GraphicsPipeline.cpp
    src/core/pipeline/ComputePipeline.cpp

    src/core/window/Window.cpp

    src/core/passes/ambientocclusion/AmbientOcclusionPass.cpp
    src/core/passes/final/FinalPass.cpp
    src/core/passes/geometry/GeometryPass.cpp
    src/core/passes/ibl/IblPass.cpp
    src/core/passes/pbr/PbrPass.cpp
    src/core/passes/imgui/ImguiPass.cpp

    src/core/scene/Mesh.cpp

    src/core/loaders/ObjLoader.cpp
    src/core/loaders/GltfLoader.cpp
    src/core/loaders/LoaderUtils.cpp

    src/core/upload/CameraUploader.cpp
    src/core/upload/LightUploader.cpp
    src/core/upload/MaterialUploader.cpp
    src/core/upload/MeshUploader.cpp

    src/core/app/GuiManager.cpp
    src/core/app/InputHandler.cpp
    src/core/app/Viewer.cpp
    
    src/core/utility/ImageLoader.cpp
)
target_include_directories(lr_core PUBLIC src)
target_link_libraries(lr_core PUBLIC
    Vulkan::Vulkan
    glfw
    glm::glm
    spdlog::spdlog
    tinyobjloader::tinyobjloader
    GPUOpen::VulkanMemoryAllocator
    imgui
    mikktspace::mikktspace
)
target_include_directories(lr_core PUBLIC ${TINYGLTF_INCLUDE_DIRS})
target_compile_definitions(lr_core PUBLIC
    LR_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders/"
    LR_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets/"
)

add_shaders(lr_core
    src/core/passes/fullscreen.vert
    src/core/passes/geometry/geometry.vert
    src/core/passes/geometry/geometry.frag
    src/core/passes/ibl/hdritocubemap.comp
    src/core/passes/ibl/irradiance.comp
    src/core/passes/ibl/prefilter.comp
    src/core/passes/final/final.frag
    src/core/passes/pbr/pbr.frag
    src/core/passes/ambientocclusion/deinterleave.comp
    src/core/passes/ambientocclusion/hbao.comp
    src/core/passes/ambientocclusion/interleave.comp
    src/core/passes/ambientocclusion/hbao_blur.comp
)

# Test executable
add_executable(renderer src/main.cpp)
target_link_libraries(renderer PRIVATE lr_core)
