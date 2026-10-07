#include "PointerCapture.hpp"

#include <imgui.h>

namespace lr
{

bool imguiCapturesPointer() { return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse; }

bool imguiCapturesKeyboard() { return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureKeyboard; }

} // namespace lr
