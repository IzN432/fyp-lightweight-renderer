#include "ObjLoader.hpp"

#include <spdlog/spdlog.h>
#include <tiny_obj_loader.h>

#include <spdlog/spdlog.h>
#include <glm/glm.hpp>

#include <spdlog/spdlog.h>
#include <cstdint>
#include <spdlog/spdlog.h>
#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <string>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <unordered_set>
#include <spdlog/spdlog.h>
#include <vector>

#include <spdlog/spdlog.h>
namespace lr
{

namespace
{

std::filesystem::path resolveTexturePath(const std::filesystem::path &objPath,
									const std::string &tex)
{
	if (tex.empty())
		return {};

	std::filesystem::path p(tex);
	if (p.is_absolute())
		return p.lexically_normal();

	return (objPath.parent_path() / p).lexically_normal();
}

std::vector<MaterialInfo> buildMaterialTable(const std::filesystem::path &objPath,
										 const std::vector<tinyobj::material_t> &materials)
{
	std::vector<MaterialInfo> out;
	out.reserve(materials.size());

	for (const tinyobj::material_t &m : materials)
	{
		MaterialInfo info{};
		info.name = m.name;

		info.baseColorFactor = glm::vec3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
		info.emissiveFactor = glm::vec3(m.emission[0], m.emission[1], m.emission[2]);
		info.roughnessFactor = m.roughness;
		info.metallicFactor = m.metallic;

		info.textures.baseColor = resolveTexturePath(objPath, m.diffuse_texname);
		info.textures.specular  = resolveTexturePath(objPath, m.specular_texname);
		info.textures.normal    = resolveTexturePath(objPath,
			!m.normal_texname.empty() ? m.normal_texname : m.bump_texname);
		info.textures.roughness = resolveTexturePath(objPath, m.roughness_texname);
		info.textures.metallic  = resolveTexturePath(objPath, m.metallic_texname);
		info.textures.emissive  = resolveTexturePath(objPath, m.emissive_texname);

		out.push_back(std::move(info));
	}

	return out;
}

std::vector<std::filesystem::path> gatherUniqueTextures(const std::vector<MaterialInfo> &materials)
{
	std::vector<std::filesystem::path> textures;
	std::unordered_set<std::string> seen;

	auto addIfValid = [&](const std::filesystem::path &p) {
		if (p.empty())
			return;
		const std::string key = p.generic_string();
		if (seen.insert(key).second)
			textures.push_back(p);
	};

	for (const MaterialInfo &m : materials)
	{
		addIfValid(m.textures.baseColor);
		addIfValid(m.textures.specular);
		addIfValid(m.textures.normal);
		addIfValid(m.textures.roughness);
		addIfValid(m.textures.metallic);
		addIfValid(m.textures.emissive);
	}

	return textures;
}

std::vector<tinyobj::material_t> loadObjMaterials(const std::filesystem::path &path)
{
	tinyobj::ObjReader reader;
	tinyobj::ObjReaderConfig config;
	config.triangulate = true;

	if (!reader.ParseFromFile(path.string(), config))
	{
		std::string msg = "ObjLoader: failed to parse materials from '" + path.string() + "'";
		if (!reader.Error().empty())
			msg += ": " + reader.Error();
		spdlog::error("Runtime error: throwing std::runtime_error");
		throw std::runtime_error(msg);
	}

	return reader.GetMaterials();
}

struct VertexKey
{
	int v  = -1;
	int vt = -1;
	int vn = -1;

	bool operator==(const VertexKey &o) const
	{
		return v == o.v && vt == o.vt && vn == o.vn;
	}
};

struct VertexKeyHash
{
	std::size_t operator()(const VertexKey &k) const
	{
		const std::size_t h1 = std::hash<int>{}(k.v);
		const std::size_t h2 = std::hash<int>{}(k.vt);
		const std::size_t h3 = std::hash<int>{}(k.vn);
		return h1 ^ (h2 << 1) ^ (h3 << 2);
	}
};

Mesh loadObjMesh(const std::filesystem::path &path, const MeshLayout &layout)
{
	tinyobj::ObjReader reader;
	tinyobj::ObjReaderConfig config;
	config.triangulate = true;

	if (!reader.ParseFromFile(path.string(), config))
	{
		std::string msg = "ObjLoader: failed to parse '" + path.string() + "'";
		if (!reader.Error().empty())
			msg += ": " + reader.Error();
		spdlog::error("Runtime error: throwing std::runtime_error");
		throw std::runtime_error(msg);
	}

	if (!reader.Warning().empty())
	{
		// Keep warnings non-fatal. Callers can route logs as needed.
	}

	const tinyobj::attrib_t &attrib = reader.GetAttrib();
	const auto &shapes = reader.GetShapes();
	const auto &materials = reader.GetMaterials();

	if (attrib.vertices.empty())
	{
	    spdlog::error("Runtime error: throwing std::runtime_error");
	    throw std::runtime_error("ObjLoader: OBJ has no vertex positions: '" + path.string() + "'");
	}

	const auto *normalDesc = layout.findPerVertexAttr("normal");
	const auto *uvDesc = layout.findPerVertexAttr("uv");
	const auto *texcoordDesc = layout.findPerVertexAttr("texcoord");
	const auto *materialIdDesc = layout.findPerFaceAttr("materialId");
	const auto *faceGroupMaterialIdDesc = layout.findFaceGroupAttr("materialId");

	const bool hasNormalAttr = (normalDesc && normalDesc->type == std::type_index(typeid(glm::vec3)));
	const bool hasUvAttr =
		(uvDesc && uvDesc->type == std::type_index(typeid(glm::vec2))) ||
		(texcoordDesc && texcoordDesc->type == std::type_index(typeid(glm::vec2)));
	const bool hasMaterialIdAttr = materialIdDesc &&
		(materialIdDesc->type == std::type_index(typeid(int32_t)) ||
		 materialIdDesc->type == std::type_index(typeid(uint32_t)));
	const bool hasFaceGroupMaterialIdAttr = faceGroupMaterialIdDesc &&
		(faceGroupMaterialIdDesc->type == std::type_index(typeid(int32_t)) ||
		 faceGroupMaterialIdDesc->type == std::type_index(typeid(uint32_t)));
	const bool needsMaterialIds = hasMaterialIdAttr || layout.faceGroupsEnabled() || hasFaceGroupMaterialIdAttr;

	const std::string uvAttrName = (uvDesc && uvDesc->type == std::type_index(typeid(glm::vec2)))
		? "uv"
		: "texcoord";

	std::vector<glm::vec3> positions;
	std::vector<glm::uvec3> faces;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> uvs;
	std::vector<int32_t> materialIds;
	positions.reserve(attrib.vertices.size() / 3);
	if (hasNormalAttr)
		normals.reserve(attrib.vertices.size() / 3);
	if (hasUvAttr)
		uvs.reserve(attrib.vertices.size() / 3);
	if (needsMaterialIds)
		materialIds.reserve(attrib.vertices.size() / 3);

	std::unordered_map<VertexKey, uint32_t, VertexKeyHash> keyToVertex;
	keyToVertex.reserve(attrib.vertices.size() / 3);

	for (const auto &shape : shapes)
	{
		size_t idxOffset = 0;
		for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
		{
			const uint8_t fv = shape.mesh.num_face_vertices[f];
			if (fv != 3)
			{
			    spdlog::error("Runtime error: throwing std::runtime_error");
			    throw std::runtime_error("ObjLoader: non-triangle face after triangulation in '" + path.string() + "'");
			}

			glm::uvec3 tri{};
			for (uint32_t v = 0; v < 3; ++v)
			{
				const tinyobj::index_t idx = shape.mesh.indices[idxOffset + v];
				const VertexKey key{idx.vertex_index, idx.texcoord_index, idx.normal_index};
				if (key.v < 0)
				{
				    spdlog::error("Runtime error: throwing std::runtime_error");
				    throw std::runtime_error("ObjLoader: invalid vertex index in '" + path.string() + "'");
				}

				auto it = keyToVertex.find(key);
				if (it == keyToVertex.end())
				{
					const size_t base = static_cast<size_t>(key.v) * 3;
					if (base + 2 >= attrib.vertices.size())
					{
					    spdlog::error("Runtime error: throwing std::runtime_error");
					    throw std::runtime_error("ObjLoader: vertex index out of bounds in '" + path.string() + "'");
					}

					positions.emplace_back(attrib.vertices[base + 0],
										   attrib.vertices[base + 1],
										   attrib.vertices[base + 2]);

					if (hasNormalAttr)
					{
						glm::vec3 n(0.0f);
						if (key.vn >= 0)
						{
							const size_t nBase = static_cast<size_t>(key.vn) * 3;
							if (nBase + 2 < attrib.normals.size())
								n = glm::vec3(attrib.normals[nBase + 0],
										  attrib.normals[nBase + 1],
										  attrib.normals[nBase + 2]);
						}
						normals.push_back(n);
					}

					if (hasUvAttr)
					{
						glm::vec2 uv(0.0f);
						if (key.vt >= 0)
						{
							const size_t uvBase = static_cast<size_t>(key.vt) * 2;
							if (uvBase + 1 < attrib.texcoords.size())
								uv = glm::vec2(attrib.texcoords[uvBase + 0],
										       attrib.texcoords[uvBase + 1]);
						}
						uvs.push_back(uv);
					}

					const uint32_t newIndex = static_cast<uint32_t>(positions.size() - 1);
					keyToVertex.emplace(key, newIndex);
					tri[v] = newIndex;
				}
				else
				{
					tri[v] = it->second;
				}
			}

			faces.push_back(tri);
			if (needsMaterialIds)
			{
				int32_t mat = -1;
				if (f < shape.mesh.material_ids.size())
					mat = shape.mesh.material_ids[f];
				if (mat >= static_cast<int32_t>(materials.size()))
					mat = -1;
				materialIds.push_back(mat);
			}
			idxOffset += fv;
		}
	}

	Mesh mesh(layout);
	mesh.setVertexCount(static_cast<uint32_t>(positions.size()));
	mesh.setFaceCount(static_cast<uint32_t>(faces.size()));
	mesh.positions = std::move(positions);
	mesh.faces = std::move(faces);

	if (hasNormalAttr)
		mesh.setPerVertexArray<glm::vec3>("normal", normals);

	if (hasUvAttr)
		mesh.setPerVertexArray<glm::vec2>(uvAttrName, uvs);

	if (hasMaterialIdAttr)
	{
		if (materialIdDesc->type == std::type_index(typeid(int32_t)))
		{
			mesh.setPerFaceArray<int32_t>("materialId", materialIds);
		}
		else
		{
			std::vector<uint32_t> mats;
			mats.reserve(materialIds.size());
			for (int32_t m : materialIds)
				mats.push_back(m < 0 ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(m));
			mesh.setPerFaceArray<uint32_t>("materialId", mats);
		}
	}

	if (layout.faceGroupsEnabled())
	{
		bool hasInvalidMaterial = false;
		for (int32_t materialId : materialIds)
		{
			if (materialId < 0)
			{
				hasInvalidMaterial = true;
				break;
			}
		}

		const uint32_t validGroupCount = static_cast<uint32_t>(materials.size());
		const uint32_t invalidGroupIndex = validGroupCount;
		const uint32_t faceGroupCount = hasInvalidMaterial
			? invalidGroupIndex + 1u
			: (validGroupCount > 0u ? validGroupCount : 1u);

		mesh.setFaceGroupCount(faceGroupCount);
		mesh.faceGroups.resize(mesh.faceCount(), invalidGroupIndex);
		for (uint32_t faceIndex = 0; faceIndex < mesh.faceCount(); ++faceIndex)
		{
			const int32_t materialId = (faceIndex < materialIds.size()) ? materialIds[faceIndex] : -1;
			mesh.faceGroups[faceIndex] = (materialId < 0)
				? invalidGroupIndex
				: static_cast<uint32_t>(materialId);
		}

		if (hasFaceGroupMaterialIdAttr)
		{
			if (faceGroupMaterialIdDesc->type == std::type_index(typeid(int32_t)))
			{
				std::vector<int32_t> groupMaterialIds(faceGroupCount, -1);
				for (uint32_t materialIndex = 0; materialIndex < materials.size(); ++materialIndex)
					groupMaterialIds[materialIndex] = static_cast<int32_t>(materialIndex);
				mesh.setFaceGroupAttributeArray<int32_t>("materialId", groupMaterialIds);
			}
			else
			{
				std::vector<uint32_t> groupMaterialIds(faceGroupCount, std::numeric_limits<uint32_t>::max());
				for (uint32_t materialIndex = 0; materialIndex < materials.size(); ++materialIndex)
					groupMaterialIds[materialIndex] = materialIndex;
				mesh.setFaceGroupAttributeArray<uint32_t>("materialId", groupMaterialIds);
			}
		}

	}

	return mesh;
}

}  // namespace

MeshSequence ObjLoader::load(const std::filesystem::path &path,
							 const MeshLayout &layout) const
{
	if (path.empty())
	{
	    throw std::invalid_argument("ObjLoader: empty mesh path");
	}

	const std::string ext = path.extension().string();
	if (ext != ".obj" && ext != ".OBJ")
	{
	    spdlog::error("Runtime error: throwing std::runtime_error");
	    throw std::runtime_error(
			"ObjLoader: unsupported extension '" + ext + "' for file '" + path.string() + "'");
	}

	MeshSequence seq;
	seq.frames.push_back(loadObjMesh(path, layout));
	seq.frameRate = 0.0f;

	const auto materials = loadObjMaterials(path);
	seq.materials = buildMaterialTable(path, materials);
	seq.textureFiles = gatherUniqueTextures(seq.materials);
	return seq;
}

}  // namespace lr
