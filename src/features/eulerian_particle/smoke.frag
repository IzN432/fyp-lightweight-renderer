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
layout(set = 0, binding = 3) uniform sampler3D temperatureVolume;

layout(push_constant) uniform SmokeRenderPC
{
    vec4 boundsMin;
    vec4 boundsMax;
    // Endpoints of the smoke's colour ramp: warmColor where the sampled temperature reaches
    // warmTemperature and above, coolColor at coolTemperature and below.
    vec4 coolColor;
    vec4 warmColor;
    // activeExtent / allocatedExtent. The volume is allocated at the largest grid the simulation
    // allows but only the origin corner of it is simulated, so the box maps to that sub-region
    // rather than to the whole texture. All ones when the two match.
    vec4 uvScale;
    float densityScale;
    float coolTemperature;
    float warmTemperature;
    uint sampleCount;
    uint enabled;
    uint frameIndex;
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

// Interleaved gradient noise. Unlike the sine hash above it decorrelates neighbouring pixels, which
// matters for the output dither: a correlated pattern would trade contours for visible clumping.
float interleavedGradientNoise(vec2 pixel)
{
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

// One LSB of the 8-bit sRGB-encoded target, expressed in the linear space the shader writes in.
// The colour attachment is an _SRGB format, so the hardware encodes after this shader runs; a flat
// 1/255 would be far too small in the darks, which is exactly where the plume's banding appears.
// This is d(linear)/d(encoded) at the current value, i.e. the linear width of one output code.
vec3 linearQuantisationStep(vec3 linearColor)
{
    vec3 encoded = sqrt(max(linearColor, vec3(0.0))); // cheap sRGB-ish encode, gamma 2
    return max(2.0 * encoded, vec3(1e-3)) / 255.0;    // derivative of encoded^2, floored near black
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
    // Jittering the ray's starting phase hides step-count striping. Varying it per frame keeps the
    // residual from freezing into a fixed screen-space pattern.
    float t = nearT + fract(randomOffset(gl_FragCoord.xy) + 0.618034 * float(pc.frameIndex & 1023u)) * stepLength;
    vec3 boxSize = max(pc.boundsMax.xyz - pc.boundsMin.xyz, vec3(1e-6));
    // Guarded so a degenerate ramp (warm <= cool) collapses to the warm end instead of dividing
    // by zero, which would otherwise show up as NaN-black pixels across the whole plume.
    float span = max(pc.warmTemperature - pc.coolTemperature, 1e-4);

    vec3 accumulated = vec3(0.0);
    float transmittance = 1.0;
    for (uint i = 0u; i < steps; ++i, t += stepLength)
    {
        vec3 worldPosition = origin + direction * t;
        vec3 uvw = (worldPosition - pc.boundsMin.xyz) / boxSize * pc.uvScale.xyz;
        float density = max(texture(densityVolume, uvw).r, 0.0) * pc.densityScale;
        float opacity = 1.0 - exp(-density * stepLength);
        // Shade each sample by its own temperature rather than tinting the finished ray: a ray
        // that grazes the hot base and then the cooled top has to come out as a mix of the two,
        // and a single tint applied afterwards cannot express that.
        float temperature = texture(temperatureVolume, uvw).r;
        float warmth = clamp((temperature - pc.coolTemperature) / span, 0.0, 1.0);
        accumulated += transmittance * opacity * mix(pc.coolColor.rgb, pc.warmColor.rgb, warmth);
        transmittance *= 1.0 - opacity;
        if (transmittance < 0.01)
            break;
    }

    float alpha = 1.0 - transmittance;
    if (alpha < 1e-4)
        discard;

    // The ray integral is effectively exact at this step count; the visible contours come from
    // quantising it to the 8-bit target. A sub-LSB offset breaks those contours into fine noise,
    // which the eye integrates away. Alpha is blended into the same 8-bit buffer, so it bands
    // independently of colour and needs its own offset.
    vec3  color  = accumulated / alpha;
    float noise  = interleavedGradientNoise(gl_FragCoord.xy + 5.588238 * float(pc.frameIndex & 1023u));
    vec3  dither = (noise - 0.5) * linearQuantisationStep(color);

    outColor = vec4(color + dither, alpha + (noise - 0.5) / 255.0);
}
