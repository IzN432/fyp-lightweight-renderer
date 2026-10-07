#include "core/scene/MeshComponent.hpp"

#include "core/loaders/Material.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <string>
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

MeshComponent::MeshComponent(MeshHandle meshHandle, MeshStore &meshStore, MaterialStore &materialStore)
    : Component("MeshComponent"), m_meshHandle(meshHandle), m_meshStore(&meshStore),
      m_materialStore(&materialStore)
{
    // One entry per face, so the same handle repeats constantly — collect the distinct ones in the
    // order they first appear to keep the inspector list stable across reloads.
    for (MaterialHandle handle : m_meshStore->get(m_meshHandle).faceGroups())
    {
        if (std::ranges::find(m_materialHandles, handle) == m_materialHandles.end())
        {
            m_materialHandles.push_back(handle);
        }
    }
}

void MeshComponent::onGUIImpl()
{
    const Mesh &mesh = m_meshStore->get(m_meshHandle);
    ImGui::Text("Mesh: %u vertices, %u faces", mesh.vertexCount(), mesh.faceCount());

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
    return m_meshStore->get(m_meshHandle);
}

const Mesh &MeshComponent::mesh() const
{
    return m_meshStore->get(m_meshHandle);
}

const MeshLayout &MeshComponent::layout() const
{
    return mesh().layout();
}

} // namespace lr
