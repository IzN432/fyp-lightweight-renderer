#pragma once

#include "core/scene/Component.hpp"
#include "core/scene/SceneObject.hpp"
#include "features/linear_blend_skinning/Skin.hpp"

#include <utility>

namespace lr
{

class SkinComponent : public Component
{
public:
    explicit SkinComponent(Skin skin) : Component("SkinComponent"), m_skin(std::move(skin)) {}

    Skin       &skin() { return m_skin; }
    const Skin &skin() const { return m_skin; }

    void evaluate()
    {
        m_skin.evaluate(getOwningObject().worldMatrix());
        markDirty();
    }

    void onGUIImpl() override
    {
        bool skinningEnabled = m_skin.skinningEnabled();
        if (ImGui::Checkbox("Skinning enabled", &skinningEnabled))
        {
            m_skin.setSkinningEnabled(skinningEnabled);
            markDirty();
        }
        if (!skinningEnabled)
        {
            // Worth spelling out, because the mesh usually appears to jump somewhere unrelated. Its
            // positions are stored in bind space, and the joint palette is what cancels this
            // object's world matrix out of the skinned result (see Skin::evaluate), so without
            // skinning that matrix applies for real to coordinates that never pass through it.
            ImGui::TextDisabled("Drawing bind-pose positions through this object's transform.");
        }

        ImGui::Text("Joint count: %zu", m_skin.joints().size());
        ImGui::TextDisabled("Vertex joint indices address this palette order.");

        if (!ImGui::TreeNode("Joint Palette"))
        {
            return;
        }

        for (JointIndex jointIndex = 0; jointIndex < m_skin.joints().size(); ++jointIndex)
        {
            const Joint       &joint = m_skin.joints()[jointIndex];
            const SceneObject &object = m_skin.jointObject(jointIndex);
            const std::string  name = object.name.empty() ? "Unnamed Scene Object" : object.name;
            const bool open = ImGui::TreeNodeEx(reinterpret_cast<void *>(static_cast<uintptr_t>(jointIndex) + 1),
                                                ImGuiTreeNodeFlags_SpanAvailWidth, "[%u] %s (ID %u)", jointIndex,
                                                name.c_str(), joint.sceneObject);
            if (open)
            {
                ImGui::TextDisabled("Inverse bind matrix");
                for (int row = 0; row < 4; ++row)
                {
                    ImGui::Text("% .3f  % .3f  % .3f  % .3f", joint.inverseBindMatrix[0][row],
                                joint.inverseBindMatrix[1][row], joint.inverseBindMatrix[2][row],
                                joint.inverseBindMatrix[3][row]);
                }
                ImGui::TreePop();
            }
        }
        ImGui::TreePop();
    }

private:
    Skin m_skin;
};

} // namespace lr
