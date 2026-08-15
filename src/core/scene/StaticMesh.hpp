#pragma once

#include "core/scene/Component.hpp"

#include "core/scene/Mesh.hpp"
#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"

#include <vector>

namespace lr
{

struct MaterialGUICallbacks
{
    explicit MaterialGUICallbacks(const std::string& materialName) : m_materialName(materialName) {}

    bool operator()(MaterialParam::ColorRGBA &param) const
    {
        return ImGui::ColorEdit4(m_materialName.c_str(), &param.value.x);
    }
    bool operator()(MaterialParam::ColorRGB &param) const
    {
        return ImGui::ColorEdit3(m_materialName.c_str(), &param.value.x);
    }
    bool operator()(MaterialParam::RangedFloat &param) const
    {
        float range = param.ceiling - param.floor;
        float increment = powf(10.0f, floor(log10(range / 100.0f)));
        return ImGui::DragFloat(m_materialName.c_str(), &param.value, increment, param.floor, param.ceiling);
    }
    bool operator()(MaterialParam::NormalizedFloat &param) const
    {
        return ImGui::DragFloat(m_materialName.c_str(), &param.value, 0.01f, 0.0f, 1.0f);
    }
private:
    std::string m_materialName;
};

class StaticMesh : public Component
{
private:
    Mesh m_mesh;
    // Handles into the MaterialStore this mesh's faceGroups index into — not owned here, just
    // referenced so onGUIImpl can offer them up for editing.
    std::vector<MaterialHandle> m_materialHandles;
    MaterialStore *m_materialStore;
    // Set for meshes that are an implementation detail of another component (e.g. a light's visual
    // quad, whose mesh/material are derived from that light and overwritten on every update) rather
    // than user-editable scene content — keeps them out of the Scene Hierarchy.
    bool m_hideFromGui;
public:
    explicit StaticMesh(Mesh &mesh, std::vector<MaterialHandle> materialHandles, MaterialStore &materialStore,
                         bool hideFromGui = false)
        : m_mesh(std::move(mesh)), m_materialHandles(std::move(materialHandles)),
          m_materialStore(&materialStore), m_hideFromGui(hideFromGui) {}

    void onGUIImpl() override
    {
        if (m_hideFromGui)
            return;

        ImGui::Text("Mesh: %u vertices, %u faces", m_mesh.vertexCount(), m_mesh.faceCount());

        bool changed = false;
        int matId = 0;
        for (MaterialHandle handle : m_materialHandles)
        {
            Material &mat = m_materialStore->get(handle);
            ImGui::PushID(matId++);
            ImGui::Text("Material: %s", mat.name.c_str());
            int paramId = 0;
            for (auto &[name, value] : mat.parameters)
            {
                ImGui::PushID(paramId++);
                changed |= (std::visit(MaterialGUICallbacks{name}, value));
                ImGui::PopID();
            }
            ImGui::PopID();
        }

        if (changed)
        {
            markDirty();
        }
    }

    const std::vector<MaterialHandle>& materialHandles() const { return m_materialHandles; }
    Mesh&       mesh()       { return m_mesh; }
    const Mesh& mesh() const { return m_mesh; }
    const MeshLayout& layout() const { return m_mesh.layout(); }
};

}