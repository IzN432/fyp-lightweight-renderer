#pragma once

#include "core/scene/Light.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/Transform.hpp"
#include "core/loaders/Material.hpp"

namespace lr
{

// Names of the mesh attributes / material parameters the area light visual must line up with,
// so it packs into the same shared vertex buffer and Materials SSBO as the rest of the scene
// (see GeometryPass — everything drawn in that pass shares one vertex layout and one material array).
struct AreaLightVisualConfig
{
    std::string normalAttributeName;
    std::string tangentAttributeName;
    std::string uvAttributeName;

    std::string baseDiffuseName;
    std::string baseEmissiveName;
    std::string baseRoughnessName;
    std::string baseMetallicName;
};

// (Re)builds `mesh` as a single flat quad standing in for an area light: centered on the light's
// Transform, spanning its local right/up axes scaled by `light.size` — the same basis CalcAreaLight
// in pbr.frag uses for the LTC quad — and facing the light's forward direction. All faces are tagged
// with `materialIndex`, an index into the combined Materials buffer the quad is uploaded alongside.
void buildAreaLightQuadMesh(Mesh &mesh, const Transform &transform, const AreaLight &light,
                             uint32_t materialIndex, const AreaLightVisualConfig &config);

// Builds the material that makes the quad read as "the light" rather than a lit surface: black
// diffuse (it reflects nothing of its own) with emissive = light.color * light.intensity, so its
// brightness in the final image tracks what CalcAreaLight computes for the same light.
Material buildAreaLightMaterial(const AreaLight &light, const AreaLightVisualConfig &config);

} // namespace lr
