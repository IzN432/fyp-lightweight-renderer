#pragma once

#include "core/scene/Mesh.hpp"
#include "core/overlay/PrimitiveOverlayMeshes.hpp"

// OverlayMesh defines a mesh that can be rendered as an overlay.
// There will be three main types:
// 1. custom meshes that can be uploaded as an obj file
// 2. meshes created from vertex positions
// 3. meshes that exist as constants

// This is the parent class for all three types

// It needs to be passed to OverlayMeshUploader to upload the mesh to the GPU.
// This can be done with a std::vector<OverlayMesh>, which will return a vector of
// something like GpuMeshLayout.

// For that matter, we can just use the standard Mesh
namespace lr
{

class OverlayMesh
{
public:
    static OverlayMesh create(OverlayMeshData data)
    {
        if (data.colors.empty())
        {
            data.colors.assign(data.positions.size(), glm::vec3(1.0f, 0.0f, 1.0f));
        }

        Mesh mesh;
        std::vector<uint32_t> positionIndices(data.positions.size());
        for (uint32_t i = 0; i < positionIndices.size(); ++i)
        {
            positionIndices[i] = i;
        }
        mesh.setTopology(std::move(data.positions), std::move(positionIndices), std::move(data.faces));
        mesh.setPerVertexArray<glm::vec3>("normal", data.normals);
        mesh.setPerVertexArray<glm::vec3>("color", data.colors);
        return OverlayMesh(std::move(mesh));
    }

    static const OverlayMesh &cube()
    {
        static OverlayMesh mesh = OverlayMesh::create(primitives::cube.surface);
        return mesh;
    }

    static const OverlayMesh &sphere()
    {
        static OverlayMesh mesh = OverlayMesh::create(primitives::sphere.surface);
        return mesh;
    }

    static const OverlayMesh &arrow()
    {
        static OverlayMesh mesh = OverlayMesh::create(primitives::arrow.surface);
        return mesh;
    }

    const Mesh &mesh() const { return m_mesh; }

private:
    OverlayMesh(Mesh &&mesh) : m_mesh(std::move(mesh)) {}

    Mesh m_mesh;
};

} // namespace lr
