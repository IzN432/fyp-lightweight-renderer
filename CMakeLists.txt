cmake_minimum_required(VERSION 3.29.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(FypLightweightRenderer LANGUAGES CXX C)

include(CTest)

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
find_package(nlohmann_json CONFIG REQUIRED)
find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h" REQUIRED)
find_package(libigl CONFIG REQUIRED)
# ARAP keeps libigl's geometry assembly but can use oneMKL PARDISO for its sparse global solve.
# Keep indices compatible with the engine's 32-bit mesh indices and avoid an OpenMP runtime in the
# interactive renderer; these variables are inputs to Intel's MKLConfig.cmake.
set(MKL_INTERFACE lp64)
set(MKL_THREADING sequential)
set(MKL_LINK dynamic)
find_package(MKL CONFIG REQUIRED)
find_package(ReactPhysics3D REQUIRED)
find_package(glslang CONFIG REQUIRED)
find_package(unofficial-spirv-reflect CONFIG REQUIRED)

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
    src/core/vulkan/ShaderCompiler.cpp
    src/core/vulkan/Allocator.cpp
    src/core/vulkan/CommandBuffer.cpp
    src/core/vulkan/DescriptorAllocator.cpp
    src/core/vulkan/Swapchain.cpp
    src/core/vulkan/Renderer.cpp

    src/core/framegraph/ResourceRegistry.cpp
    src/core/framegraph/FrameGraph.cpp
    src/core/framegraph/CompiledFrameGraph.cpp
    src/core/framegraph/FrameGraphDefinition.cpp
    src/core/framegraph/PassBuilder.cpp
    src/core/framegraph/PassContext.cpp
    src/core/framegraph/PassDescAdapter.cpp
    src/core/framegraph/FrameGraphTopology.cpp
    src/core/framegraph/model/GraphDefinition.cpp
    src/core/framegraph/compiler/GraphCompiler.cpp
    src/core/framegraph/compiler/FrameGraphCompiler.cpp
    src/core/framegraph/compiler/VulkanBarrierPlanner.cpp
    src/core/framegraph/compiler/ShaderInterface.cpp
    src/core/framegraph/ImageReadback.cpp

    src/core/pipeline/GraphicsPipeline.cpp
    src/core/pipeline/ComputePipeline.cpp

    src/core/window/Window.cpp

    src/core/passes/ambientocclusion/AmbientOcclusionPass.cpp
    src/core/passes/composite/CompositePass.cpp
    src/core/passes/final/FinalPass.cpp
    src/core/passes/geometry/GeometryPass.cpp
    src/core/passes/heatmap/HeatmapPass.cpp
    src/core/passes/overlaygeometry/OverlayGeometryPass.cpp
    src/core/passes/overlaylines/OverlayLine.cpp
    src/core/passes/overlaylines/OverlayLinesPass.cpp
    src/core/passes/overlaypoints/OverlayPointsPass.cpp

    src/core/passes/ibl/IblPass.cpp
    src/core/passes/pbr/PbrPass.cpp
    src/core/passes/imgui/ImguiPass.cpp

    src/core/scene/Mesh.cpp
    src/core/scene/MeshComponent.cpp
    src/core/scene/MeshStore.cpp
    src/core/scene/TransformComponent.cpp
    src/core/scene/AreaLightVisual.cpp
    src/core/scene/Scene.cpp
    src/core/scene/SceneObject.cpp
    src/core/scene/SceneManager.cpp
    src/core/scene/SceneGpu.cpp
    src/core/scene/EngineConventions.cpp

    src/core/loaders/ObjLoader.cpp
    src/core/loaders/GltfLoader.cpp
    src/core/loaders/SceneLoader.cpp
    src/core/loaders/LoaderUtils.cpp
    src/core/loaders/MaterialStore.cpp

    src/core/upload/CameraUploader.cpp
    src/core/upload/LightUploader.cpp
    src/core/upload/MaterialUploader.cpp
    src/core/upload/MeshUploader.cpp
    src/core/upload/SkinUploader.cpp

    src/core/app/GuiManager.cpp
    src/core/app/ImGuiWidgets.cpp
    src/core/app/InputHandler.cpp
    src/core/app/Viewer.cpp

    src/core/editor/VertexManager.cpp
    src/core/editor/DefaultVertexDragHandler.cpp
    src/core/editor/SceneObjectDragHandler.cpp
    src/core/editor/SceneObjectRotationHandler.cpp
    src/core/editor/SceneObjectScaleHandler.cpp
    src/core/editor/SceneObjectTransformController.cpp

    src/core/editor/command/TranslatePointsCommand.cpp
    src/core/editor/command/TranslateSceneObjectCommand.cpp
    src/core/editor/command/RotateSceneObjectCommand.cpp
    src/core/editor/command/ScaleSceneObjectCommand.cpp
    src/core/editor/command/CommandManager.cpp

    src/core/editor/selection/BoxSelectionTool.cpp
    src/core/editor/selection/SelectionManager.cpp

    src/core/editor/gizmo/TranslateGizmo.cpp
    src/core/editor/gizmo/RotateGizmo.cpp
    src/core/editor/gizmo/ScaleGizmo.cpp

    src/core/editor/camera/SphericalCameraController.cpp

    src/core/math/LinearAlgebraHelpers.cpp

    src/core/utility/ImageLoader.cpp
    src/core/utility/HeatmapColors.cpp
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
    imguizmo
    mikktspace::mikktspace
)
target_link_libraries(lr_core PRIVATE
    glslang::glslang
    glslang::glslang-default-resource-limits
    glslang::SPIRV
    unofficial::spirv-reflect
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
    src/core/passes/composite/composite.frag
    src/core/passes/pbr/pbr.frag
    src/core/passes/ambientocclusion/deinterleave.comp
    src/core/passes/ambientocclusion/hbao.comp
    src/core/passes/ambientocclusion/interleave.comp
    src/core/passes/ambientocclusion/hbao_blur.comp
    src/core/passes/overlaygeometry/overlay_geometry.vert
    src/core/passes/overlaygeometry/overlay_geometry.frag
    src/core/passes/overlaypoints/overlay_points.vert
    src/core/passes/overlaypoints/overlay_points.frag
    src/core/passes/heatmap/heatmap.vert
    src/core/passes/heatmap/heatmap.frag
    src/core/passes/overlaylines/overlay_lines.vert
    src/core/passes/overlaylines/overlay_lines.frag

)

# ARAP editor feature — physically separate from lr_core and linked by the application.
add_library(lr_arap_feature STATIC
    src/features/arap/ArapSolver.cpp
    src/features/arap/ArapDragHandler.cpp
    src/features/arap/ArapDeformCommand.cpp
    src/features/arap/ArapTool.cpp
)
target_include_directories(lr_arap_feature PUBLIC src)
target_link_libraries(lr_arap_feature PUBLIC lr_core igl::igl_core PRIVATE MKL::MKL)

# Laplace-Beltrami analysis feature. Core owns only the generic heatmap rendering/buffer path;
# this target owns the mesh operator, color generation, GUI, and stale-result tracking.
add_library(lr_laplace_beltrami_feature STATIC
    src/features/laplace_beltrami/LaplaceBeltramiOperator.cpp
    src/features/laplace_beltrami/LaplaceBeltramiTool.cpp
)
target_include_directories(lr_laplace_beltrami_feature PUBLIC src)
target_link_libraries(lr_laplace_beltrami_feature PUBLIC lr_core igl::igl_core)

add_library(lr_linear_blend_skinning_feature STATIC
    src/features/linear_blend_skinning/Skin.cpp
)
target_include_directories(lr_linear_blend_skinning_feature PUBLIC src)
target_link_libraries(lr_linear_blend_skinning_feature PUBLIC lr_core glm::glm)

add_library(lr_animation_feature STATIC
    src/features/animation/AnimationTrack.cpp
    src/features/animation/AnimationClip.cpp
    src/features/animation/AnimatorComponent.cpp
)
target_include_directories(lr_animation_feature PUBLIC src)
target_link_libraries(lr_animation_feature PUBLIC lr_core glm::glm)

add_library(lr_rigid_body_feature STATIC
    src/features/rigid_body/ColliderComponent.cpp
    src/features/rigid_body/ColliderVisual.cpp
    src/features/rigid_body/PhysicsWorld.cpp
    src/features/rigid_body/RigidBodyComponent.cpp
)
target_include_directories(lr_rigid_body_feature PUBLIC src)
target_link_libraries(lr_rigid_body_feature PUBLIC lr_core glm::glm ReactPhysics3D::ReactPhysics3D)

# Native scene persistence spans the core scene model and optional component feature libraries, so
# keep it above those libraries rather than making lr_core depend back on its own features.
add_library(lr_scene_serialization STATIC
    src/core/scene/SceneSerializer.cpp
)
target_include_directories(lr_scene_serialization PUBLIC src)
target_link_libraries(lr_scene_serialization PUBLIC lr_core lr_animation_feature
    lr_linear_blend_skinning_feature lr_rigid_body_feature nlohmann_json::nlohmann_json)

# Application executable
add_executable(renderer src/main.cpp)
target_link_libraries(renderer PRIVATE lr_core lr_arap_feature lr_laplace_beltrami_feature
    lr_linear_blend_skinning_feature lr_animation_feature lr_rigid_body_feature lr_scene_serialization
    imguiFileDialog)

# Python module — `lr` package assembled in build/python/lr (run scripts with PYTHONPATH=build/python).
# Needs a Python interpreter with nanobind installed (pip install nanobind); skipped otherwise.
option(LR_BUILD_PYTHON "Build the lr Python module" ON)
if(LR_BUILD_PYTHON)
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module)
    if(Python_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
    find_package(nanobind CONFIG QUIET)
    if(NOT nanobind_FOUND)
        message(WARNING "LR_BUILD_PYTHON: Python or nanobind not found — skipping the lr Python module")
    else()
        set(LR_PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/lr")
        nanobind_add_module(_lr src/python/LrModule.cpp)
        # The feature libraries SceneLoader (in lr_core) instantiates components from.
        target_link_libraries(_lr PRIVATE lr_core lr_animation_feature lr_linear_blend_skinning_feature
            lr_rigid_body_feature lr_scene_serialization)
        # $<0:> keeps multi-config generators from appending a per-config subdirectory.
        set_target_properties(_lr PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${LR_PYTHON_PACKAGE_DIR}$<0:>"
            RUNTIME_OUTPUT_DIRECTORY "${LR_PYTHON_PACKAGE_DIR}$<0:>")

        # Python only searches the extension's own directory for its DLL dependencies. The
        # interpreter's own DLL is already loaded by whichever Python imports the module.
        add_custom_command(TARGET _lr POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<LIST:FILTER,$<TARGET_RUNTIME_DLLS:_lr>,EXCLUDE,python[0-9]+\\.dll$>" "${LR_PYTHON_PACKAGE_DIR}"
            COMMAND_EXPAND_LISTS)

        file(GLOB LR_PYTHON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/lr/*.py")
        add_custom_target(lr_python ALL
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LR_PYTHON_SOURCES} "${LR_PYTHON_PACKAGE_DIR}"
            DEPENDS _lr
            COMMENT "Assembling the lr Python package")

        if(BUILD_TESTING)
            # Both open short-lived windows, so they need a GPU and a display.
            add_test(NAME python.bindings
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_bindings.py")
            add_test(NAME python.spinning_torus
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/python/spinning_torus.py"
                    --frames 120)
            add_test(NAME python.pipeline_state
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/python/pipeline_state.py"
                    --frames 120)
            add_test(NAME python.failures
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_failures.py")
            # Both assert on their own results (readback vs numpy; compile counts) and, via the
            # regex below, fail on any validation error, e.g. destroying a buffer still in use.
            add_test(NAME python.gpu_instancing
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/python/gpu_instancing.py"
                    --frames 250)
            add_test(NAME python.live_edit
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/python/live_edit.py"
                    --frames 300)
            # Asserts every frame read its own uniform data (149/150 frames didn't, before per-frame copies).
            add_test(NAME python.per_frame_data
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/python/per_frame_data.py")
            # Drives the orbit camera with injected mouse input and asserts it orbited and zoomed.
            add_test(NAME python.interactive
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/python/interactive.py"
                    --scripted)
            set_tests_properties(python.bindings python.spinning_torus python.pipeline_state python.failures
                python.gpu_instancing python.live_edit python.per_frame_data python.interactive PROPERTIES
                ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
                # nanobind prints leak reports to stderr without failing the process.
                FAIL_REGULAR_EXPRESSION "nanobind: leaked|\\[error\\]")
            # Scenes loaded with the engine's loaders, read from Python (CPU only).
            add_test(NAME python.scene
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_scene.py")
            set_tests_properties(python.scene PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
            # A Python renderer drawing an engine-loaded glTF scene.
            add_test(NAME python.scene_viewer
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/python/scene_viewer.py"
                    --frames 120)
            set_tests_properties(python.scene_viewer PROPERTIES
                ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
                FAIL_REGULAR_EXPRESSION "nanobind: leaked|\\[error\\]")
            # The engine's passes as building blocks: contracts, rendering, camera sync, lifetimes; and a
            # deferred renderer assembled from them with a Python pass in the middle.
            add_test(NAME python.blocks
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_blocks.py")
            add_test(NAME python.deferred_blocks
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/python/deferred_blocks.py"
                    --frames 120)
            # Adds, edits and removes lights while running, checking the GPU light buffer after each step.
            add_test(NAME python.light_editor
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/python/light_editor.py"
                    --scripted)
            # A frame loop owned by the script: viewer.step() with a numpy simulation between frames.
            add_test(NAME python.step_loop
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/python/step_loop.py"
                    --frames 200)
            set_tests_properties(python.blocks python.deferred_blocks python.light_editor python.step_loop PROPERTIES
                ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
                FAIL_REGULAR_EXPRESSION "nanobind: leaked|\\[error\\]")

            # The hand-written stubs must match the module (names, parameters) and type-check the examples.
            add_test(NAME python.stubs
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_stubs.py")
            set_tests_properties(python.stubs PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
            add_test(NAME python.typecheck
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/typecheck.py")
            set_tests_properties(python.typecheck PROPERTIES SKIP_RETURN_CODE 77)

            # Provokes errors on purpose (including one logged validation error), and asserts on each.
            add_test(NAME python.error_handling
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/python/error_handling.py")
            set_tests_properties(python.error_handling PROPERTIES
                ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
                FAIL_REGULAR_EXPRESSION "nanobind: leaked")
        endif()
    endif()
endif()

if(BUILD_TESTING)
    add_executable(mesh_buffer_sync_tests
        tests/upload/MeshBufferSyncTests.cpp
        src/core/scene/Mesh.cpp
        src/core/editor/VertexManager.cpp
    )
    target_include_directories(mesh_buffer_sync_tests PRIVATE src)
    target_link_libraries(mesh_buffer_sync_tests PRIVATE Vulkan::Vulkan glm::glm)
    add_test(NAME mesh.buffer_sync COMMAND mesh_buffer_sync_tests)

    # Opt-in headless Vulkan check: build/run explicitly on a machine with a Vulkan device.
    # Keep hardware requirements out of the ordinary CPU-only CTest suite.
    add_executable(mesh_gpu_sync_smoke EXCLUDE_FROM_ALL tests/upload/MeshGpuSyncSmoke.cpp)
    target_link_libraries(mesh_gpu_sync_smoke PRIVATE lr_core)

    add_executable(scene_hierarchy_tests
        tests/scene/SceneHierarchyTests.cpp
    )
    target_link_libraries(scene_hierarchy_tests PRIVATE lr_core)
    add_test(NAME scene.hierarchy COMMAND scene_hierarchy_tests)

    add_executable(scene_serialization_tests
        tests/scene/SceneSerializationTests.cpp
    )
    target_link_libraries(scene_serialization_tests PRIVATE lr_scene_serialization)
    add_test(NAME scene.serialization COMMAND scene_serialization_tests)

    add_executable(linear_blend_skinning_joint_tests
        tests/linear_blend_skinning/JointTests.cpp
    )
    target_link_libraries(linear_blend_skinning_joint_tests PRIVATE lr_core lr_linear_blend_skinning_feature)
    add_test(NAME linear_blend_skinning.joint COMMAND linear_blend_skinning_joint_tests)

    add_executable(gltf_skin_loader_tests
        tests/linear_blend_skinning/GltfSkinLoaderTests.cpp
    )
    target_link_libraries(gltf_skin_loader_tests PRIVATE lr_core lr_linear_blend_skinning_feature
        lr_animation_feature)
    target_compile_definitions(gltf_skin_loader_tests PRIVATE
        LR_TEST_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/"
        LR_SAMPLE_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets/samples/models/"
    )
    add_test(NAME linear_blend_skinning.gltf_loader COMMAND gltf_skin_loader_tests)

    add_executable(animation_tests
        tests/animation/AnimationTests.cpp
    )
    target_link_libraries(animation_tests PRIVATE lr_animation_feature)
    add_test(NAME animation.model COMMAND animation_tests)

    add_executable(rigid_body_collider_tests
        tests/rigid_body/ColliderTests.cpp
    )
    target_link_libraries(rigid_body_collider_tests PRIVATE lr_rigid_body_feature)
    add_test(NAME rigid_body.collider COMMAND rigid_body_collider_tests)

    # Opt-in CPU benchmark for comparing ARAP solver implementations. It deliberately is not a
    # CTest: wall-clock performance is machine-dependent and should be run explicitly in Release.
    add_executable(arap_profile EXCLUDE_FROM_ALL tests/arap/ArapProfile.cpp)
    target_link_libraries(arap_profile PRIVATE lr_arap_feature)
    target_compile_definitions(arap_profile PRIVATE
        LR_SAMPLE_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets/samples/models/"
    )

    add_executable(shader_compiler_tests
        tests/vulkan/ShaderCompilerTests.cpp
    )
    target_link_libraries(shader_compiler_tests PRIVATE lr_core)
    target_compile_definitions(shader_compiler_tests PRIVATE
        LR_SHADER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/core/passes"
    )
    add_test(NAME vulkan.shader_compiler COMMAND shader_compiler_tests)

    add_executable(shader_interface_tests
        tests/framegraph/ShaderInterfaceTests.cpp
    )
    target_link_libraries(shader_interface_tests PRIVATE lr_core)
    add_test(NAME framegraph.shader_interface COMMAND shader_interface_tests)

    add_executable(framegraph_model_tests
        tests/framegraph/GraphModelTests.cpp
        src/core/framegraph/model/GraphDefinition.cpp
        src/core/framegraph/compiler/GraphCompiler.cpp
    )
    target_include_directories(framegraph_model_tests PRIVATE src)
    add_test(NAME framegraph.model COMMAND framegraph_model_tests)

    add_executable(framegraph_topology_tests
        tests/framegraph/FrameGraphTopologyTests.cpp
        src/core/framegraph/FrameGraphDefinition.cpp
        src/core/framegraph/PassBuilder.cpp
        src/core/framegraph/PassDescAdapter.cpp
        src/core/framegraph/FrameGraphTopology.cpp
        src/core/framegraph/model/GraphDefinition.cpp
        src/core/framegraph/compiler/GraphCompiler.cpp
        src/core/framegraph/compiler/VulkanBarrierPlanner.cpp
        src/core/scene/Mesh.cpp
    )
    target_include_directories(framegraph_topology_tests PRIVATE src)
    target_link_libraries(framegraph_topology_tests PRIVATE Vulkan::Vulkan glm::glm)
    add_test(NAME framegraph.topology COMMAND framegraph_topology_tests)
endif()
