#version 450

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

layout(set = 0, binding = 1) uniform sampler3D densityVolume;
layout(set = 0, binding = 2) uniform sampler2D sceneDepth;

layout(push_constant) uniform SmokeRenderPC
{
    vec4 boundsMin;
    vec4 boundsMax;
    vec4 color;
    float densityScale;
    uint sampleCount;
    uint enabled;
    uint padding;
} pc;

layout(location = 0) out vec4 outColor;

vec3 reconstructWorld(vec2 uv, float depth)
{
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 world = cameraUbo.invView * cameraUbo.invProj * clip;
    return world.xyz / world.w;
}

bool intersectBox(vec3 origin, vec3 direction, out float nearT, out float farT)
{
    vec3 safeDirection = mix(vec3(1e-7), direction, greaterThan(abs(direction), vec3(1e-7)));
    vec3 inverseDirection = 1.0 / safeDirection;
    vec3 t0 = (pc.boundsMin.xyz - origin) * inverseDirection;
    vec3 t1 = (pc.boundsMax.xyz - origin) * inverseDirection;
    vec3 lo = min(t0, t1);
    vec3 hi = max(t0, t1);
    nearT = max(max(lo.x, lo.y), lo.z);
    farT = min(min(hi.x, hi.y), hi.z);
    return farT > max(nearT, 0.0);
}

float randomOffset(vec2 pixel)
{
    return fract(sin(dot(pixel, vec2(12.9898, 78.233))) * 43758.5453);
}

void main()
{
    if (pc.enabled == 0u || pc.densityScale <= 0.0)
        discard;

    // Starting at the near plane works for both perspective and orthographic cameras and matches
    // the renderer's clip range (smoke between the eye and the near plane is not visible).
    vec3 origin = reconstructWorld(inUV, 0.0);
    vec3 farPoint = reconstructWorld(inUV, 1.0);
    vec3 direction = normalize(farPoint - origin);

    float nearT;
    float farT;
    if (!intersectBox(origin, direction, nearT, farT))
        discard;

    nearT = max(nearT, 0.0);
    float depth = texture(sceneDepth, inUV).r;
    if (depth < 1.0 - 1e-6)
    {
        vec3 opaquePoint = reconstructWorld(inUV, depth);
        farT = min(farT, dot(opaquePoint - origin, direction));
    }
    if (farT <= nearT)
        discard;

    uint steps = max(pc.sampleCount, 1u);
    float stepLength = (farT - nearT) / float(steps);
    float t = nearT + randomOffset(gl_FragCoord.xy) * stepLength;
    vec3 boxSize = max(pc.boundsMax.xyz - pc.boundsMin.xyz, vec3(1e-6));

    vec3 accumulated = vec3(0.0);
    float transmittance = 1.0;
    for (uint i = 0u; i < steps; ++i, t += stepLength)
    {
        vec3 worldPosition = origin + direction * t;
        vec3 uvw = (worldPosition - pc.boundsMin.xyz) / boxSize;
        float density = max(texture(densityVolume, uvw).r, 0.0) * pc.densityScale;
        float opacity = 1.0 - exp(-density * stepLength);
        accumulated += transmittance * opacity * pc.color.rgb;
        transmittance *= 1.0 - opacity;
        if (transmittance < 0.01)
            break;
    }

    float alpha = 1.0 - transmittance;
    if (alpha < 1e-4)
        discard;
    outColor = vec4(accumulated / alpha, alpha);
}
