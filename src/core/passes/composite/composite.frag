#version 450

#include "../utility/tonemap.glslh"

// CompositePass: the engine's final image without the editor overlays. Where GeometryPass drew nothing
// (gbufferDepth == 1) shows the environment cubemap; elsewhere the HDR input image. Both are Reinhard
// tone mapped — the same as final.frag (see tonemap.glslh).

layout(location = 0) in vec2 inUV;

layout(set = 0, binding = 0) uniform CameraUbo
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invView;
    mat4 invProj;
    vec4 position;
} cameraUbo;

layout(set = 0, binding = 1) uniform samplerCube skybox;
layout(set = 0, binding = 2) uniform sampler2D gbufferDepth;
layout(set = 0, binding = 3) uniform sampler2D hdrInput;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 hdr;
    if (isBackground(texture(gbufferDepth, inUV).r))
    {
        hdr = texture(skybox, viewRayWorld(inUV, cameraUbo.view, cameraUbo.proj)).rgb;
    }
    else
    {
        hdr = texture(hdrInput, inUV).rgb;
    }
    outColor = vec4(reinhard(hdr), 1.0);
}
