#pragma once

#include "core/scene/Component.hpp"

#include "core/scene/Mesh.hpp"
#include "core/loaders/Material.hpp"

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
    MeshLayout& m_layout;
    std::vector<Material> m_materials;
public:
    explicit StaticMesh(Mesh &mesh, MeshLayout &layout, std::vector<Material>& materials)
        : m_mesh(std::move(mesh)), m_layout(layout), m_materials(std::move(materials)) {}

    void onGUIImpl() override
    {
        ImGui::Text("Mesh: %u vertices, %u faces", m_mesh.vertexCount(), m_mesh.faceCount());
        
        bool changed = false;
        int matId = 0;
        for (auto &mat : m_materials)
        {
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
            notifyChanged();
        }
    }

    const std::vector<Material>& materials() const { return m_materials; }
    const Mesh& mesh() const { return m_mesh; }
    const MeshLayout& layout() const { return m_layout; }
};

}