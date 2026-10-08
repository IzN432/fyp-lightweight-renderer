#include "core/scene/MeshComponent.hpp"

#include "core/loaders/Material.hpp"
#include "core/scene/EngineConventions.hpp"
#include "core/scene/SceneObject.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace lr
{
namespace
{

constexpr std::array<std::string_view, 7> materialParameterOrder = {
    conventions::baseDiffuse,   conventions::baseEmissive, conventions::baseRoughness,
    conventions::baseMetallic,  conventions::alphaCutoff,  conventions::alphaBlend,
    conventions::doubleSided,
};

size_t materialParameterRank(const std::string_view name)
{
    const auto found = std::ranges::find(materialParameterOrder, name);
    return static_cast<size_t>(std::distance(materialParameterOrder.begin(), found));
}

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
        if (materialName == conventions::alphaBlend || materialName == conventions::doubleSided)
        {
            bool enabled = param.value >= 0.5f;
            if (!ImGui::Checkbox(materialName.c_str(), &enabled))
            {
                return false;
            }

            param.value = enabled ? 1.0f : 0.0f;
            return true;
        }

        return ImGui::DragFloat(materialName.c_str(), &param.value, 0.01f, 0.0f, 1.0f);
    }

    std::string materialName;
};

} // namespace

MeshComponent::MeshComponent(MeshHandle meshHandle, MeshStore &meshStore, MaterialStore &materialStore)
    : Component("MeshComponent"), m_meshHandle(meshHandle), m_meshStore(&meshStore),
      m_materialStore(&materialStore)
{
    collectMaterialHandles();
}

void MeshComponent::collectMaterialHandles()
{
    // One entry per face, so the same handle repeats constantly — collect the distinct ones in the
    // order they first appear to keep the inspector list stable across reloads.
    m_materialHandles.clear();
    for (MaterialHandle handle : m_meshStore->get(m_meshHandle).faceGroups())
    {
        if (std::ranges::find(m_materialHandles, handle) == m_materialHandles.end())
        {
            m_materialHandles.push_back(handle);
        }
    }
}

std::unique_ptr<ComponentValues> MeshComponent::copyValues() const
{
    return std::make_unique<ComponentValueSnapshot<MeshId>>(m_meshStore->idOf(m_meshHandle));
}

ComponentValuesAdder MeshComponent::valuesAdder() const
{
    // The stores outlive any one scene — SceneManager owns them and a load clears them rather than
    // replacing them — so capturing them is safe even across a load. Whether the copied *mesh* is
    // still in there is a separate question, which addPasted answers by id.
    return [meshStore = m_meshStore, materialStore = m_materialStore](SceneObject            &object,
                                                                     const ComponentValues &values) {
        addPasted(object, *meshStore, *materialStore, values);
    };
}

void MeshComponent::addPasted(SceneObject &object, MeshStore &meshStore, MaterialStore &materialStore,
                              const ComponentValues &values)
{
    const std::optional<MeshHandle> handle = meshStore.find(componentValuesAs<MeshId>(values));
    // The copied mesh is not in this store — the usual way being a copy taken before a scene load
    // cleared it. There is no geometry to point at, so nothing is added.
    if (!handle)
    {
        return;
    }
    MeshComponent &added = object.addComponent<MeshComponent>(*handle, meshStore, materialStore);
    // The object only joins the gathered geometry once the editor registers it (see
    // ComponentPasteService); this is what then asks for the re-pack that uploads it. Geometry
    // alone, because a re-pack re-uploads the materials on its way through.
    added.markDirty(Aspect::Geometry);
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

        std::vector<std::string> parameterNames;
        parameterNames.reserve(material.parameters.size());
        for (const auto &[name, value] : material.parameters)
        {
            parameterNames.push_back(name);
        }
        std::ranges::sort(parameterNames, [](const std::string &left, const std::string &right) {
            const size_t leftRank  = materialParameterRank(left);
            const size_t rightRank = materialParameterRank(right);
            return leftRank != rightRank ? leftRank < rightRank : left < right;
        });

        for (const std::string &name : parameterNames)
        {
            ImGui::PushID(name.c_str());
            changed |= std::visit(MaterialGUICallbacks{name}, material.parameters.at(name));
            ImGui::PopID();
        }
        ImGui::PopID();
    }

    if (changed)
    {
        markDirty(Aspect::Materials);
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
