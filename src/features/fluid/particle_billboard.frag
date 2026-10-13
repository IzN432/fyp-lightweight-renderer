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
layout(location = 2) flat in vec4 inColorAndType;
layout(location = 3) flat in float inRadius;
layout(location = 0) out vec4 outColor;

void main()
{
    float radiusSquared = dot(inSphereCoord, inSphereCoord);
    if (radiusSquared > 1.0)
    {
        discard;
    }

    // The camera looks down -Z, so the surface nearest the camera is offset toward +Z from the
    // centre. The shading normal points in that same direction.
    float nearZ = sqrt(max(0.0, 1.0 - radiusSquared));
    vec3 normal = vec3(inSphereCoord, nearZ);
    vec3 viewPosition = inViewCenter + normal * inRadius;
    vec4 clipPosition = cameraUbo.proj * vec4(viewPosition, 1.0);
    float particleDepth = clipPosition.z / clipPosition.w;

    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float opaqueDepth = texelFetch(gbufferDepth, pixel, 0).r;
    if (particleDepth > opaqueDepth + 1e-5)
    {
        discard;
    }

    gl_FragDepth = particleDepth;
    // The normal is view-space, so the world-space key direction has to be rotated to match or the
    // light would swing around with the camera.
    vec3 lightDirection = normalize(mat3(cameraUbo.view) * vec3(-0.35, 0.7, 0.6));
    float diffuse = 0.3 + 0.7 * max(dot(normal, lightDirection), 0.0);
    float rim = pow(1.0 - normal.z, 2.0);
    outColor = vec4(inColorAndType.rgb * diffuse + vec3(0.12) * rim, 1.0);
}
