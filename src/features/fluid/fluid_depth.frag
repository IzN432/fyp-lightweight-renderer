#version 450

layout(set = 0, binding = 0) uniform CameraUbo
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invView;
    mat4 invProj;
    vec4 cameraPosition;
} cameraUbo;

layout(set = 0, binding = 2) uniform sampler2D gbufferDepth;

layout(location = 0) in vec2 inSphereCoord;
layout(location = 1) in vec3 inViewCenter;
layout(location = 2) flat in float inRadius;
layout(location = 3) flat in uint inType;
layout(location = 0) out float outLinearDepth;

void main()
{
    float radiusSquared = dot(inSphereCoord, inSphereCoord);
    if (inType != 0u || inRadius <= 0.0 || radiusSquared > 1.0)
        discard;

    float nearZ = sqrt(max(0.0, 1.0 - radiusSquared));
    vec3 viewPosition = inViewCenter + vec3(inSphereCoord, nearZ) * inRadius;
    vec4 clipPosition = cameraUbo.proj * vec4(viewPosition, 1.0);
    if (clipPosition.w <= 0.0)
        discard;

    float particleDepth = clipPosition.z / clipPosition.w;
    if (particleDepth < 0.0 || particleDepth > 1.0)
        discard;

    // Sampled with normalized UVs, not texelFetch: if the fluid targets are ever sized
    // differently from the G-buffer, this still lines up instead of silently misaligning.
    vec2 sceneUv = gl_FragCoord.xy / vec2(textureSize(gbufferDepth, 0));
    float opaqueDepth = texture(gbufferDepth, sceneUv).r;
    if (particleDepth > opaqueDepth + 1e-5)
        discard;

    // Zero doubles as the attachment clear value and as the "no fluid here" sentinel every
    // consumer tests for, so a fragment must never write a non-positive distance. Anything at or
    // behind the eye plane is rejected rather than written as an ambiguous zero.
    float linearDepth = -viewPosition.z;
    if (linearDepth <= 0.0)
        discard;

    gl_FragDepth = particleDepth;
    outLinearDepth = linearDepth;
}
