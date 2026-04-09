set(SRC
  ImGuiFileDialog.cpp
  ImGuiFileDialog.h
  ImGuiFileDialogConfig.h
)

add_library(imguiFileDialog STATIC ${SRC})
target_link_libraries(imguiFileDialog PUBLIC imgui)
target_compile_definitions(imguiFileDialog PRIVATE USE_STD_FILESYSTEM)
target_include_directories(imguiFileDialog PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
