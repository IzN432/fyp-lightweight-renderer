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
layout(set = 0, binding = 3) uniform LiquidParams
{
    vec4 absorptionAndRefraction;
    vec4 filterAndMaterial;
    vec4 tintAndFresnel;
    ivec4 filterRadius;
} params;

layout(location = 0) in vec2 inSphereCoord;
layout(location = 1) in vec3 inViewCenter;
layout(location = 2) flat in float inRadius;
layout(location = 3) flat in uint inType;
layout(location = 0) out float outThickness;

void main()
{
    float radiusSquared = dot(inSphereCoord, inSphereCoord);
    if (inType != 0u || inRadius <= 0.0 || radiusSquared > 1.0)
        discard;

    float halfChord = sqrt(max(0.0, 1.0 - radiusSquared)) * inRadius;
    vec3 nearestPoint = inViewCenter + vec3(inSphereCoord * inRadius, halfChord);
    vec4 clipPosition = cameraUbo.proj * vec4(nearestPoint, 1.0);
    if (clipPosition.w <= 0.0)
        discard;
    float particleDepth = clipPosition.z / clipPosition.w;
    // Normalized UVs keep this aligned with the G-buffer even at a differing target resolution.
    vec2 sceneUv = gl_FragCoord.xy / vec2(textureSize(gbufferDepth, 0));
    float opaqueDepth = texture(gbufferDepth, sceneUv).r;
    if (particleDepth < 0.0 || particleDepth > opaqueDepth + 1e-5)
        discard;

    // Clip the far half of the sphere at opaque scene geometry. Testing only the front surface
    // prevents hidden particles, but without this clip a visible particle intersecting a wall
    // would still absorb light through its complete diameter.
    float nearLinearDepth = -nearestPoint.z;
    float farLinearDepth = -(inViewCenter.z - halfChord);
    if (opaqueDepth < 1.0 - 1e-6)
    {
        vec4 opaqueView = cameraUbo.invProj * vec4(sceneUv * 2.0 - 1.0, opaqueDepth, 1.0);
        float opaqueLinearDepth = -(opaqueView.z / opaqueView.w);
        farLinearDepth = min(farLinearDepth, opaqueLinearDepth);
    }

    outThickness = max(farLinearDepth - nearLinearDepth, 0.0) * params.filterAndMaterial.z;
}
