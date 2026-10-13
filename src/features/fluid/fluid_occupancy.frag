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
layout(set = 0, binding = 3) uniform sampler2D fluidFrontDepth;
layout(set = 0, binding = 4) uniform sampler2D fluidBackDepth;
layout(set = 0, binding = 5, r32ui) uniform uimage2D occupancyImage;

layout(location = 0) in vec2 inSphereCoord;
layout(location = 1) in vec3 inViewCenter;
layout(location = 2) flat in float inRadius;
layout(location = 3) flat in uint inType;

const int kSliceCount = 32;

void main()
{
    float radiusSquared = dot(inSphereCoord, inSphereCoord);
    if (inType != 0u || inRadius <= 0.0 || radiusSquared > 1.0)
        discard;

    float halfChord = sqrt(max(0.0, 1.0 - radiusSquared)) * inRadius;
    vec3 nearPoint = inViewCenter + vec3(inSphereCoord * inRadius, halfChord);
    vec4 nearClip = cameraUbo.proj * vec4(nearPoint, 1.0);
    if (nearClip.w <= 0.0)
        discard;

    float projectedNear = nearClip.z / nearClip.w;
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(gbufferDepth, 0);
    vec2 sceneUv = gl_FragCoord.xy / vec2(size);
    float opaqueDepth = texture(gbufferDepth, sceneUv).r;
    if (projectedNear < 0.0 || projectedNear > opaqueDepth + 1e-5)
        discard;

    float nearDepth = -nearPoint.z;
    float farDepth = -(inViewCenter.z - halfChord);
    if (opaqueDepth < 1.0 - 1e-6)
    {
        vec4 opaqueView = cameraUbo.invProj * vec4(sceneUv * 2.0 - 1.0, opaqueDepth, 1.0);
        farDepth = min(farDepth, -(opaqueView.z / opaqueView.w));
    }

    float spanFront = texelFetch(fluidFrontDepth, pixel, 0).r;
    float spanBack = texelFetch(fluidBackDepth, pixel, 0).r;
    float span = spanBack - spanFront;
    if (spanFront <= 0.0 || span <= 1e-6)
        discard;

    nearDepth = clamp(nearDepth, spanFront, spanBack);
    farDepth = clamp(farDepth, spanFront, spanBack);
    if (farDepth <= nearDepth)
        discard;

    int first = clamp(int(floor((nearDepth - spanFront) * float(kSliceCount) / span)), 0,
                      kSliceCount - 1);
    int lastExclusive = clamp(int(ceil((farDepth - spanFront) * float(kSliceCount) / span)),
                              first + 1, kSliceCount);

    uint lowerMask = first == 0 ? 0xffffffffu : (0xffffffffu << first);
    uint upperMask = lastExclusive == kSliceCount ? 0xffffffffu : ((1u << lastExclusive) - 1u);
    imageAtomicOr(occupancyImage, pixel, lowerMask & upperMask);
}
