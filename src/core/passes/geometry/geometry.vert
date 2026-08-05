#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUv;

layout(set = 0, binding = 0) uniform CameraUbo
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invView;
    mat4 invProj;
    vec3 cameraPosition;
    float padding;
} cameraUbo;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outTangent;
layout(location = 3) out vec2 outUv;

// Mirrors the push_constant block in geometry.frag — model is this draw's mesh-to-world matrix
// (identity for meshes whose vertices are already baked into world space, e.g. area light quads).
// primitiveIdOffset is unused here but must stay in the layout so both stages agree on offsets.
layout(push_constant) uniform PC
{
    mat4 model;
    uint primitiveIdOffset;
} pc;

void main()
{
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    outWorldPos = worldPos.xyz;

    mat3 normalMatrix = transpose(inverse(mat3(pc.model)));
    outNormal = normalize(normalMatrix * inNormal);
    outTangent = vec4(normalize(mat3(pc.model) * inTangent.xyz), inTangent.w);

    outUv = inUv;
    gl_Position = cameraUbo.viewProj * worldPos;
}
