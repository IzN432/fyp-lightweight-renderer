#version 450
#extension GL_ARB_shader_viewport_layer_array : require

#include "../utility/skinning.glslh"

layout(location = 0) in vec3 inPosition;

layout(set = 0, binding = 0) uniform ShadowData
{
    mat4 lightViewProj[16];
    uvec4 header;
} shadow;

layout(push_constant) uniform PC
{
    mat4 model;
    uint paletteOffset;
    uint skinEnabled;
} pc;

void main()
{
    vec3 position = inPosition;
    if (pc.skinEnabled != 0)
        position = (calculateSkinMatrix(gl_VertexIndex, pc.paletteOffset) * vec4(position, 1.0)).xyz;
    uint shadowIndex = gl_InstanceIndex;
    gl_Layer = int(shadowIndex);
    gl_Position = shadow.lightViewProj[shadowIndex] * pc.model * vec4(position, 1.0);
}
