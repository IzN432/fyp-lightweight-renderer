set(IMGUI_SRC
    imgui.h
    imgui_internal.h
    imconfig.h
    imgui.cpp
    imgui_demo.cpp
    imgui_draw.cpp
    imgui_tables.cpp
    imgui_widgets.cpp

    imgui_impl_vulkan.h
    imgui_impl_vulkan.cpp

    imgui_impl_glfw.h
    imgui_impl_glfw.cpp
)

add_library(imgui STATIC ${IMGUI_SRC})
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)
target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
