#version 450

layout(location = 0) in vec2 inUV;

layout(set = 0, binding = 0) uniform CameraUbo
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invView;
    mat4 invProj;
    vec4 cameraPosition;
} cameraUbo;
layout(set = 0, binding = 1) uniform sampler2D fluidDepth;
layout(set = 0, binding = 2) uniform sampler2D fluidNormal;
layout(set = 0, binding = 3) uniform sampler2D fluidThickness;
layout(set = 0, binding = 4) uniform sampler2D sceneColor;
layout(set = 0, binding = 5) uniform sampler2D sceneDepth;
layout(set = 0, binding = 6) uniform samplerCube environmentMap;
layout(set = 0, binding = 7) uniform LiquidParams
{
    vec4 absorptionAndRefraction;
    vec4 filterAndMaterial;
    vec4 tintAndFresnel;
    ivec4 filterRadius;
} params;

layout(location = 0) out vec4 outColor;

// Beyond this optical path length the absorption and displacement terms are visually saturated,
// so clamping here keeps the controls responsive instead of pinning them.
const float kMaxThickness = 6.0;
// Converts the reprojected Snell displacement into the same perceptual range the old
// slope-scaled offset occupied, so refractionStrength keeps its existing tuning scale.
const float kRefractionGain = 50.0;
// Halving steps used to pull a refracted sample back off a foreground occluder. Four steps shrink
// the offset to 1/16 before giving up, which is visually indistinguishable from no refraction.
const int kRefractionSearchSteps = 4;
// Fractional slack when comparing scene distance against the liquid surface, covering the
// depth-buffer round-trip error that grows with distance.
const float kOcclusionTolerance = 1e-3;

vec3 viewRay(vec2 uv)
{
    vec4 farPoint = cameraUbo.invProj * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    return normalize(farPoint.xyz / farPoint.w);
}

// Positive view-space distance to the opaque scene, or a huge value where nothing was drawn so
// the sky never counts as an occluder.
float sceneLinearDepth(vec2 uv)
{
    float depth = texture(sceneDepth, uv).r;
    if (depth >= 1.0 - 1e-6)
        return 1e30;
    vec4 viewPosition = cameraUbo.invProj * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return -(viewPosition.z / viewPosition.w);
}

vec3 backgroundAt(vec2 uv)
{
    float depth = texture(sceneDepth, uv).r;
    if (depth < 1.0 - 1e-6)
        return texture(sceneColor, uv).rgb;
    vec3 worldDirection = mat3(cameraUbo.invView) * viewRay(uv);
    return texture(environmentMap, worldDirection).rgb;
}

void main()
{
    float depth = texture(fluidDepth, inUV).r;
    vec4 normalSample = texture(fluidNormal, inUV);
    if (depth <= 0.0 || normalSample.a <= 0.0)
    {
        outColor = vec4(0.0);
        return;
    }

    vec3 normal = normalize(normalSample.xyz);
    vec3 incident = viewRay(inUV);
    vec3 viewDirection = -incident;
    // Clamped so a dense pile of overlapping spheres cannot push the exponentials below into
    // saturation, where thickness would stop modulating absorption and displacement entirely.
    float thickness = clamp(texture(fluidThickness, inUV).r, 0.0, kMaxThickness);

    // Snell's law gives the true bend for this index of refraction, so the IOR drives both the
    // Fresnel mix below and the displacement here from one value. refractionStrength stays as a
    // deliberate artistic exaggeration on top of the physical result rather than a substitute.
    float eta = 1.0 / max(params.filterAndMaterial.w, 1.0); // entering the liquid from air
    vec3 refractedDirection = refract(incident, normal, eta);

    vec2 refractionOffset = vec2(0.0);
    // refract() returns zero under total internal reflection. Entering from air (eta < 1) cannot
    // reach that case, so this guard is insurance for a future back-face exit pass rather than a
    // branch the front surface takes.
    if (dot(refractedDirection, refractedDirection) >= 1e-8)
    {
        // Walk along the refracted ray for the optical path length, then reproject. The
        // displacement therefore follows the IOR and the thickness instead of the surface slope
        // alone, and vanishes where the liquid is thin.
        // fluidDepth stores positive view-space distance along -Z, while `incident` is a unit
        // ray, so scale it onto that depth plane rather than treating depth as a ray length.
        vec3 entryPoint = incident * (depth / max(-incident.z, 1e-6));
        vec4 exitClip = cameraUbo.proj * vec4(entryPoint + refractedDirection * thickness, 1.0);
        if (exitClip.w > 0.0)
            refractionOffset = (exitClip.xy / exitClip.w) * 0.5 + 0.5 - inUV;
    }

    // Screen-space refraction can only sample what the camera already saw, so a bent lookup may
    // land on geometry standing in FRONT of the liquid and smear that foreground object through
    // the body of the fluid. Walk the offset back toward the unbent sample until it lands on
    // something at least as far away as the surface. Shortening the offset rather than dropping
    // it outright keeps neighbouring pixels from flipping between bent and unbent, which would
    // otherwise read as a hard seam along the occluder's edge.
    vec2 fullOffset = refractionOffset * params.absorptionAndRefraction.w * kRefractionGain;
    vec2 refractedUv = clamp(inUV + fullOffset, vec2(0.001), vec2(0.999));
    for (int step = 0; step < kRefractionSearchSteps; ++step)
    {
        // Relative tolerance, because reconstructing distance from a float depth buffer loses
        // precision in proportion to distance. A fluid surface resting directly on opaque
        // geometry is the common case, and a fixed epsilon would misread that contact as an
        // occluder far from the camera and cancel refraction exactly where it is wanted.
        if (sceneLinearDepth(refractedUv) >= depth * (1.0 - kOcclusionTolerance))
            break;
        fullOffset *= 0.5;
        refractedUv = clamp(inUV + fullOffset, vec2(0.001), vec2(0.999));
    }

    vec3 transmitted = backgroundAt(refractedUv);
    vec3 transmittance = exp(-params.absorptionAndRefraction.xyz * thickness);
    transmitted = transmitted * transmittance + params.tintAndFresnel.rgb * (1.0 - transmittance);

    vec3 reflectedView = reflect(incident, normal);
    vec3 reflectedWorld = mat3(cameraUbo.invView) * reflectedView;
    vec3 reflected = texture(environmentMap, reflectedWorld).rgb;
    float nDotV = clamp(dot(normal, viewDirection), 0.0, 1.0);
    float fresnel = params.tintAndFresnel.w + (1.0 - params.tintAndFresnel.w) * pow(1.0 - nDotV, 5.0);

    outColor = vec4(mix(transmitted, reflected, fresnel), 1.0);
}
