#include "core/editor/ErrorReporter.hpp"

#include <imgui.h>

#include <utility>

namespace lr
{

void ErrorReporter::report(std::string title, std::string message)
{
    m_pending.push_back({std::move(title), std::move(message)});
}

void ErrorReporter::drawPopup()
{
    if (!m_current && !m_pending.empty())
    {
        m_current = std::move(m_pending.front());
        m_pending.pop_front();
        ImGui::OpenPopup("Editor Error");
    }

    if (!ImGui::BeginPopupModal("Editor Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        return;
    }
    ImGui::TextUnformatted(m_current->title.c_str());
    ImGui::Separator();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(m_current->message.c_str());
    ImGui::PopTextWrapPos();
    if (ImGui::Button("OK"))
    {
        m_current.reset();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

} // namespace lr
