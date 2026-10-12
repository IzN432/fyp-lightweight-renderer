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

layout(push_constant) uniform ParticleBillboardPC
{
    vec4 colorAndRadius;
} pc;

layout(location = 0) in vec2 inSphereCoord;
layout(location = 1) in vec3 inViewCenter;
layout(location = 0) out vec4 outColor;

void main()
{
    float radiusSquared = dot(inSphereCoord, inSphereCoord);
    if (radiusSquared > 1.0)
    {
        discard;
    }

    // The camera looks down -Z, so positive view-space Z is the sphere surface nearest the camera.
    vec3 normal = vec3(inSphereCoord, sqrt(max(0.0, 1.0 - radiusSquared)));
    vec3 viewPosition = inViewCenter + normal * pc.colorAndRadius.w;
    vec4 clipPosition = cameraUbo.proj * vec4(viewPosition, 1.0);
    float particleDepth = clipPosition.z / clipPosition.w;

    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float opaqueDepth = texelFetch(gbufferDepth, pixel, 0).r;
    if (particleDepth > opaqueDepth + 1e-5)
    {
        discard;
    }

    gl_FragDepth = particleDepth;
    vec3 lightDirection = normalize(vec3(-0.35, 0.7, 0.6));
    float diffuse = 0.3 + 0.7 * max(dot(normal, lightDirection), 0.0);
    float rim = pow(1.0 - normal.z, 2.0);
    outColor = vec4(pc.colorAndRadius.rgb * diffuse + vec3(0.12) * rim, 0.9);
}
