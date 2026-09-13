#include "core/scene/MeshComponent.hpp"

#include "core/loaders/Material.hpp"

#include <imgui.h>

#include <cmath>
#include <string>
#include <utility>
#include <variant>

namespace lr
{
namespace
{

struct MaterialGUICallbacks
{
    explicit MaterialGUICallbacks(const std::string &materialName) : materialName(materialName) {}

    bool operator()(MaterialParam::ColorRGBA &param) const
    {
        return ImGui::ColorEdit4(materialName.c_str(), &param.value.x);
    }

    bool operator()(MaterialParam::ColorRGB &param) const
    {
        return ImGui::ColorEdit3(materialName.c_str(), &param.value.x);
    }

    bool operator()(MaterialParam::RangedFloat &param) const
    {
        const float range     = param.ceiling - param.floor;
        const float increment = std::pow(10.0f, std::floor(std::log10(range / 100.0f)));
        return ImGui::DragFloat(materialName.c_str(), &param.value, increment, param.floor, param.ceiling);
    }

    bool operator()(MaterialParam::NormalizedFloat &param) const
    {
        return ImGui::DragFloat(materialName.c_str(), &param.value, 0.01f, 0.0f, 1.0f);
    }

    std::string materialName;
};

} // namespace

MeshComponent::MeshComponent(Mesh &mesh, std::vector<MaterialHandle> materialHandles, MaterialStore &materialStore,
                             bool hideFromGui)
    : Component("MeshComponent"), m_mesh(std::move(mesh)), m_materialHandles(std::move(materialHandles)),
      m_materialStore(&materialStore), m_hideFromGui(hideFromGui)
{}

void MeshComponent::onGUIImpl()
{
    if (m_hideFromGui)
    {
        return;
    }

    ImGui::Text("Mesh: %u vertices, %u faces", m_mesh.vertexCount(), m_mesh.faceCount());

    bool changed = false;
    int  materialId = 0;
    for (MaterialHandle handle : m_materialHandles)
    {
        Material &material = m_materialStore->get(handle);
        ImGui::PushID(materialId++);
        ImGui::Text("Material: %s", material.name.c_str());

        int parameterId = 0;
        for (auto &[name, value] : material.parameters)
        {
            ImGui::PushID(parameterId++);
            changed |= std::visit(MaterialGUICallbacks{name}, value);
            ImGui::PopID();
        }
        ImGui::PopID();
    }

    if (changed)
    {
        markDirty();
    }
}

const std::vector<MaterialHandle> &MeshComponent::materialHandles() const
{
    return m_materialHandles;
}

Mesh &MeshComponent::mesh()
{
    return m_mesh;
}

const Mesh &MeshComponent::mesh() const
{
    return m_mesh;
}

const MeshLayout &MeshComponent::layout() const
{
    return m_mesh.layout();
}

} // namespace lr
