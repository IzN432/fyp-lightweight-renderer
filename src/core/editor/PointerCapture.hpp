#pragma once

namespace lr
{

// Whether ImGui has claimed the pointer / keyboard this frame. Both answer false when there is no
// ImGui context at all, so a Viewer running without a GUI behaves as if nothing is capturing.
// Shared so that every place which has to yield to the UI asks the same question the same way.
bool imguiCapturesPointer();
bool imguiCapturesKeyboard();

} // namespace lr
