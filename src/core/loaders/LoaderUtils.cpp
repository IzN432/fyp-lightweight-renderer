#include "core/loaders/LoaderUtils.hpp"

#include <mikktspace.h>

namespace lr
{

namespace
{
    
/**
 * Static functions for the Mikktspace tangent generation.
 */
int msGetNumFaces(const SMikkTSpaceContext *context)
{
    auto *data = static_cast<const MeshData *>(context->m_pUserData);
    return static_cast<int>(data->faces.size());
}

int msGetNumVerticesOfFace(const SMikkTSpaceContext *context, const int faceNum)
{
    return 3; // Assuming all faces are triangles
}

void msGetPosition(const SMikkTSpaceContext *context, float outPos[], const int faceNum, const int vertNum)
{
    auto *data = static_cast<const MeshData *>(context->m_pUserData);
    const glm::vec3 &pos = data->positions[data->faces[faceNum][vertNum]];
    outPos[0] = pos.x;
    outPos[1] = pos.y;
    outPos[2] = pos.z;
}

void msGetNormal(const SMikkTSpaceContext *context, float outNormal[], const int faceNum, const int vertNum)
{
    auto *data = static_cast<const MeshData *>(context->m_pUserData);
    const glm::vec3 &normal = data->normals[data->faces[faceNum][vertNum]];
    outNormal[0] = normal.x;
    outNormal[1] = normal.y;
    outNormal[2] = normal.z;
}

void msGetTexcoord(const SMikkTSpaceContext *context, float outTexcoord[], const int faceNum, const int vertNum)
{
    auto *data = static_cast<const MeshData *>(context->m_pUserData);
    const glm::vec2 &uvs = data->uvs[data->faces[faceNum][vertNum]];
    outTexcoord[0] = uvs.x;
    outTexcoord[1] = uvs.y;
}

void msSetTSpace(const SMikkTSpaceContext *context, const float tangent[], const float sign, const int faceNum, const int vertNum)
{
    auto *data = static_cast<MeshData *>(context->m_pUserData);
    glm::vec4 &tangentOut = data->tangents[data->faces[faceNum][vertNum]];
    tangentOut.x = tangent[0];
    tangentOut.y = tangent[1];
    tangentOut.z = tangent[2];
    tangentOut.w = sign;
}

} // namespace

void generateTangents(MeshData &meshData)
{
    SMikkTSpaceInterface interface = {};
    interface.m_getNumFaces = msGetNumFaces;
    interface.m_getNumVerticesOfFace = msGetNumVerticesOfFace;
    interface.m_getPosition = msGetPosition;
    interface.m_getNormal = msGetNormal;
    interface.m_getTexCoord = msGetTexcoord;
    interface.m_setTSpaceBasic = msSetTSpace;

    SMikkTSpaceContext context = {};
    context.m_pInterface = &interface;
    context.m_pUserData = &meshData;

    genTangSpaceDefault(&context);
}

} // namespace lr