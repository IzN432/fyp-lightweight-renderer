#include "Scene.hpp"

#include <imgui.h>

namespace lr
{

void Scene::onGUI()
{
    ImGui::SeparatorText("Scene Objects");

    int id = 0;
    for (auto &object : m_sceneObjects)
    {
        ImGui::PushID(id++);
        object->onGUI();
        ImGui::PopID();
    }
}

}  // namespace lr
