#pragma once

#include <cmath>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace lr
{

struct OverlayMeshData
{
    std::vector<glm::vec3>  positions;
    std::vector<glm::vec3>  normals;
    std::vector<glm::uvec3> faces;
    std::vector<glm::vec3>  colors;
};

struct OverlayLineMeshData
{
    std::vector<glm::vec3>  positions;
    std::vector<glm::uvec2> edges;
    std::vector<glm::vec3>  colors;
};

// A single canonical editor primitive. Filled and outlined representations deliberately
// keep separate vertex domains: surfaces may duplicate positions for flat normals, while
// outlines generally need only unique corners or a few selected silhouette curves.
struct OverlayPrimitiveData
{
    OverlayMeshData     surface;
    OverlayLineMeshData outline;
};

namespace primitives
{

inline OverlayMeshData makeCubeSurface()
{
    return {
    .positions =
        {
            // Front  (+Z)
            {-0.5f, -0.5f, 0.5f},
            {0.5f, -0.5f, 0.5f},
            {0.5f, 0.5f, 0.5f},
            {-0.5f, 0.5f, 0.5f},
            // Back   (-Z)
            {0.5f, -0.5f, -0.5f},
            {-0.5f, -0.5f, -0.5f},
            {-0.5f, 0.5f, -0.5f},
            {0.5f, 0.5f, -0.5f},
            // Right  (+X)
            {0.5f, -0.5f, 0.5f},
            {0.5f, -0.5f, -0.5f},
            {0.5f, 0.5f, -0.5f},
            {0.5f, 0.5f, 0.5f},
            // Left   (-X)
            {-0.5f, -0.5f, -0.5f},
            {-0.5f, -0.5f, 0.5f},
            {-0.5f, 0.5f, 0.5f},
            {-0.5f, 0.5f, -0.5f},
            // Top    (+Y)
            {-0.5f, 0.5f, 0.5f},
            {0.5f, 0.5f, 0.5f},
            {0.5f, 0.5f, -0.5f},
            {-0.5f, 0.5f, -0.5f},
            // Bottom (-Y)
            {-0.5f, -0.5f, -0.5f},
            {0.5f, -0.5f, -0.5f},
            {0.5f, -0.5f, 0.5f},
            {-0.5f, -0.5f, 0.5f},
        },
    .normals =
        {
            // Front
            {0.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 1.0f},
            // Back
            {0.0f, 0.0f, -1.0f},
            {0.0f, 0.0f, -1.0f},
            {0.0f, 0.0f, -1.0f},
            {0.0f, 0.0f, -1.0f},
            // Right
            {1.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            // Left
            {-1.0f, 0.0f, 0.0f},
            {-1.0f, 0.0f, 0.0f},
            {-1.0f, 0.0f, 0.0f},
            {-1.0f, 0.0f, 0.0f},
            // Top
            {0.0f, 1.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            // Bottom
            {0.0f, -1.0f, 0.0f},
            {0.0f, -1.0f, 0.0f},
            {0.0f, -1.0f, 0.0f},
            {0.0f, -1.0f, 0.0f},
        },
    .faces =
        {
            {0, 1, 2},
            {0, 2, 3}, // Front
            {4, 5, 6},
            {4, 6, 7}, // Back
            {8, 9, 10},
            {8, 10, 11}, // Right
            {12, 13, 14},
            {12, 14, 15}, // Left
            {16, 17, 18},
            {16, 18, 19}, // Top
            {20, 21, 22},
            {20, 22, 23}, // Bottom
        },
    };
}

inline OverlayLineMeshData makeCubeOutline()
{
    return {
        .positions = {
            {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f},
            {0.5f, 0.5f, -0.5f},   {-0.5f, 0.5f, -0.5f},
            {-0.5f, -0.5f, 0.5f},  {0.5f, -0.5f, 0.5f},
            {0.5f, 0.5f, 0.5f},    {-0.5f, 0.5f, 0.5f},
        },
        .edges = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0},
            {4, 5}, {5, 6}, {6, 7}, {7, 4},
            {0, 4}, {1, 5}, {2, 6}, {3, 7},
        },
    };
}

inline const OverlayPrimitiveData cube = {
    .surface = makeCubeSurface(),
    .outline = makeCubeOutline(),
};

inline OverlayMeshData makeSphereSurface(int latCount = 12, int lonCount = 16)
{
    OverlayMeshData data;
    for (int lat = 0; lat <= latCount; ++lat)
    {
        float theta = static_cast<float>(lat) * glm::pi<float>() / static_cast<float>(latCount);
        for (int lon = 0; lon <= lonCount; ++lon)
        {
            float     phi = static_cast<float>(lon) * 2.0f * glm::pi<float>() / static_cast<float>(lonCount);
            glm::vec3 pos = {
                std::sin(theta) * std::cos(phi),
                std::cos(theta),
                std::sin(theta) * std::sin(phi),
            };
            data.positions.push_back(pos);
            data.normals.push_back(pos); // unit sphere: normal == position
        }
    }
    for (int lat = 0; lat < latCount; ++lat)
    {
        for (int lon = 0; lon < lonCount; ++lon)
        {
            uint32_t v0 = static_cast<uint32_t>(lat * (lonCount + 1) + lon);
            uint32_t v1 = v0 + 1;
            uint32_t v2 = v0 + static_cast<uint32_t>(lonCount + 1);
            uint32_t v3 = v2 + 1;
            data.faces.push_back({v0, v2, v1});
            data.faces.push_back({v1, v2, v3});
        }
    }
    return data;
}

inline OverlayLineMeshData makeSphereOutline(int segments = 64)
{
    OverlayLineMeshData data;
    const float         angleStep = glm::two_pi<float>() / static_cast<float>(segments);

    // Two orthogonal great circles: XY followed by YZ.
    for (int circle = 0; circle < 2; ++circle)
    {
        const uint32_t base = static_cast<uint32_t>(data.positions.size());
        for (int segment = 0; segment < segments; ++segment)
        {
            const float angle = angleStep * static_cast<float>(segment);
            data.positions.push_back(circle == 0 ? glm::vec3(std::cos(angle), std::sin(angle), 0.0f)
                                                 : glm::vec3(0.0f, std::cos(angle), std::sin(angle)));
            data.edges.push_back({base + static_cast<uint32_t>(segment),
                                  base + static_cast<uint32_t>((segment + 1) % segments)});
        }
    }
    return data;
}

inline OverlayPrimitiveData makeSphere(int latCount = 12, int lonCount = 16, int outlineSegments = 64)
{
    return {
        .surface = makeSphereSurface(latCount, lonCount),
        .outline = makeSphereOutline(outlineSegments),
    };
}

inline OverlayMeshData makeArrowSurface(int segments = 8)
{
    OverlayMeshData data;
    // Arrow along +Y: shaft y=[0, 0.7] r=0.05; cone y=[0.7, 1.0] base r=0.12
    const float twoPi      = 2.0f * glm::pi<float>();
    const float shaftR     = 0.05f;
    const float shaftTop   = 0.7f;
    const float coneR      = 0.12f;
    const float coneTop    = 1.0f;
    const float coneH      = coneTop - shaftTop;
    const float coneSlantL = std::sqrt(coneH * coneH + coneR * coneR);

    auto addRing = [&](float y, float r, glm::vec3 flatNormal, bool useRadialNormal) {
        for (int i = 0; i < segments; ++i)
        {
            float     phi = twoPi * static_cast<float>(i) / static_cast<float>(segments);
            glm::vec3 pos = {r * std::cos(phi), y, r * std::sin(phi)};
            glm::vec3 n   = useRadialNormal ? glm::normalize(glm::vec3(pos.x, 0.0f, pos.z)) : flatNormal;
            data.positions.push_back(pos);
            data.normals.push_back(n);
        }
    };

    // Shaft side rings (radial normals)
    uint32_t shaftBotBase = 0;
    addRing(0.0f, shaftR, {}, true);
    uint32_t shaftTopBase = static_cast<uint32_t>(segments);
    addRing(shaftTop, shaftR, {}, true);

    for (int i = 0; i < segments; ++i)
    {
        uint32_t b0 = shaftBotBase + static_cast<uint32_t>(i);
        uint32_t b1 = shaftBotBase + static_cast<uint32_t>((i + 1) % segments);
        uint32_t t0 = shaftTopBase + static_cast<uint32_t>(i);
        uint32_t t1 = shaftTopBase + static_cast<uint32_t>((i + 1) % segments);
        data.faces.push_back({b0, t0, b1});
        data.faces.push_back({b1, t0, t1});
    }

    // Shaft bottom cap (normal: 0,-1,0)
    uint32_t shaftBotCapBase = static_cast<uint32_t>(2 * segments);
    addRing(0.0f, shaftR, {0, -1, 0}, false);
    uint32_t shaftBotCenter = static_cast<uint32_t>(data.positions.size());
    data.positions.push_back({0.0f, 0.0f, 0.0f});
    data.normals.push_back({0, -1, 0});
    for (int i = 0; i < segments; ++i)
    {
        uint32_t v0 = shaftBotCapBase + static_cast<uint32_t>(i);
        uint32_t v1 = shaftBotCapBase + static_cast<uint32_t>((i + 1) % segments);
        data.faces.push_back({v0, v1, shaftBotCenter});
    }

    // Cone sides (outward slant normals)
    uint32_t coneBaseRing = static_cast<uint32_t>(data.positions.size());
    for (int i = 0; i < segments; ++i)
    {
        float     phi = twoPi * static_cast<float>(i) / static_cast<float>(segments);
        glm::vec3 pos = {coneR * std::cos(phi), shaftTop, coneR * std::sin(phi)};
        glm::vec3 n   = glm::normalize(
            glm::vec3(std::cos(phi) * coneH / coneSlantL, coneR / coneSlantL, std::sin(phi) * coneH / coneSlantL));
        data.positions.push_back(pos);
        data.normals.push_back(n);
    }
    uint32_t apex = static_cast<uint32_t>(data.positions.size());
    data.positions.push_back({0.0f, coneTop, 0.0f});
    data.normals.push_back({0, 1, 0});
    for (int i = 0; i < segments; ++i)
    {
        uint32_t v0 = coneBaseRing + static_cast<uint32_t>(i);
        uint32_t v1 = coneBaseRing + static_cast<uint32_t>((i + 1) % segments);
        data.faces.push_back({v0, apex, v1});
    }

    // Cone base cap (normal: 0,-1,0)
    uint32_t coneCapBase = static_cast<uint32_t>(data.positions.size());
    for (int i = 0; i < segments; ++i)
    {
        float phi = twoPi * static_cast<float>(i) / static_cast<float>(segments);
        data.positions.push_back({coneR * std::cos(phi), shaftTop, coneR * std::sin(phi)});
        data.normals.push_back({0, -1, 0});
    }
    uint32_t coneCapCenter = static_cast<uint32_t>(data.positions.size());
    data.positions.push_back({0.0f, shaftTop, 0.0f});
    data.normals.push_back({0, -1, 0});
    for (int i = 0; i < segments; ++i)
    {
        uint32_t v0 = coneCapBase + static_cast<uint32_t>(i);
        uint32_t v1 = coneCapBase + static_cast<uint32_t>((i + 1) % segments);
        data.faces.push_back({v0, v1, coneCapCenter});
    }

    return data;
}

inline OverlayLineMeshData makeArrowOutline(int segments = 8)
{
    OverlayLineMeshData data;
    const float         angleStep = glm::two_pi<float>() / static_cast<float>(segments);
    constexpr float     shaftRadius = 0.05f;
    constexpr float     shaftTop    = 0.7f;
    constexpr float     coneRadius  = 0.12f;

    const auto addRing = [&](float y, float radius) {
        const uint32_t base = static_cast<uint32_t>(data.positions.size());
        for (int segment = 0; segment < segments; ++segment)
        {
            const float angle = angleStep * static_cast<float>(segment);
            data.positions.push_back({radius * std::cos(angle), y, radius * std::sin(angle)});
            data.edges.push_back({base + static_cast<uint32_t>(segment),
                                  base + static_cast<uint32_t>((segment + 1) % segments)});
        }
        return base;
    };

    const uint32_t shaftBottom = addRing(0.0f, shaftRadius);
    const uint32_t shaftTopRing = addRing(shaftTop, shaftRadius);
    const uint32_t coneBase = addRing(shaftTop, coneRadius);
    const uint32_t apex = static_cast<uint32_t>(data.positions.size());
    data.positions.push_back({0.0f, 1.0f, 0.0f});

    for (int segment = 0; segment < segments; ++segment)
    {
        const uint32_t i = static_cast<uint32_t>(segment);
        data.edges.push_back({shaftBottom + i, shaftTopRing + i});
        data.edges.push_back({shaftTopRing + i, coneBase + i});
        data.edges.push_back({coneBase + i, apex});
    }
    return data;
}

inline OverlayPrimitiveData makeArrow(int segments = 8)
{
    return {
        .surface = makeArrowSurface(segments),
        .outline = makeArrowOutline(segments),
    };
}

inline const OverlayPrimitiveData sphere = makeSphere();
inline const OverlayPrimitiveData arrow  = makeArrow();

// Unit XZ rectangle used by finite plane colliders and other editor overlays.
inline const OverlayPrimitiveData plane = {
    .outline = {
        .positions = {
            {-0.5f, 0.0f, -0.5f},
            {0.5f, 0.0f, -0.5f},
            {0.5f, 0.0f, 0.5f},
            {-0.5f, 0.0f, 0.5f},
        },
        .edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}},
    },
};

} // namespace primitives

} // namespace lr
