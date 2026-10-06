#pragma once

#include "core/loaders/Material.hpp"
#include "core/loaders/SceneLoader.hpp"
#include "core/scene/AreaLightVisual.hpp"
#include "core/scene/Mesh.hpp"

#include <cstdint>
#include <string>
#include <vector>

// The names the engine's loaders write into meshes and materials, and its passes read back
// (GeometryPass / geometry.frag and the GPU material layout). Anything that produces or consumes
// scene data for those passes — the C++ renderer, the Python module — takes them from here, so the
// two sides can't disagree.
namespace lr::conventions
{

// Per-vertex mesh attributes.
inline constexpr const char *normalAttribute  = "normal";  // vec3
inline constexpr const char *tangentAttribute = "tangent"; // vec4: xyz tangent, w handedness
inline constexpr const char *uvAttribute      = "uv";      // vec2

// Material textures (RGBA8; base colour and emissive are sRGB).
inline constexpr const char *baseColorTexture         = "baseColorTexture";
inline constexpr const char *normalTexture            = "normalTexture";
inline constexpr const char *metallicRoughnessTexture = "metallicRoughnessTexture"; // glTF: G roughness, B metallic
inline constexpr const char *emissiveTexture          = "emissiveTexture";

// Material parameters.
inline constexpr const char *baseDiffuse   = "baseDiffuse";   // RGBA colour
inline constexpr const char *baseEmissive  = "baseEmissive";  // RGB colour
inline constexpr const char *baseRoughness = "baseRoughness"; // [0, 1]
inline constexpr const char *baseMetallic  = "baseMetallic";  // [0, 1]
inline constexpr const char *alphaCutoff   = "alphaCutoff";   // glTF MASK threshold; 0 disables masking
inline constexpr const char *doubleSided   = "doubleSided";   // 0 or 1
inline constexpr const char *alphaBlend    = "alphaBlend";    // 1 = glTF BLEND (TransparentPass draws it)

// MaterialStore slots reserved up front: growing it would mean resizing the GPU material buffer and
// texture arrays, i.e. rebuilding descriptor sets (see MaterialStore.hpp).
inline constexpr uint32_t materialCapacity = 256;

// Loader configurations that write the names above.
GltfLoaderConfig  gltfLoaderConfig();
SceneLoaderConfig sceneLoaderConfig();

// What a MaterialStore slot holds before a material is assigned: black, fully rough, non-metallic.
Material defaultMaterial();

// The per-vertex attributes packed, interleaved in this order, into the vertex buffer GeometryPass
// reads at binding 1 (binding 0 holds positions alone): normal, tangent, uv.
std::vector<std::string> geometryVertexAttributes();

// GeometryPass's vertex input: position (binding 0, location 0, vec3); normal (binding 1, location 1,
// vec3), tangent (binding 1, location 2, vec4) and uv (binding 1, location 3, vec2).
GpuMeshLayout geometryMeshLayout();

// The materials SSBO layout geometry.frag reads (64-byte stride): baseDiffuse vec4 @0, baseEmissive vec3
// @16, baseRoughness float @32, baseMetallic float @36, alphaCutoff float @40, doubleSided float @44,
// alphaBlend float @48 (see utility/material.glslh); plus one texture array per texture name above.
GpuMaterialLayout materialLayout();

// Names the light-visual quads (see AreaLightVisual.hpp) are built with, so they pack into the same
// vertex buffer and materials SSBO as loaded meshes.
AreaLightVisualConfig areaLightVisualConfig();

} // namespace lr::conventions
