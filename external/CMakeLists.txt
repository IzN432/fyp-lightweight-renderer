add_subdirectory(imgui)
add_subdirectory(ImGuiFileDialog)

add_library(imguizmo STATIC ImGuizmo.cpp)
target_link_libraries(imguizmo PUBLIC imgui)
target_include_directories(imguizmo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
