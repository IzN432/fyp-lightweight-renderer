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

layout(location = 0) out vec4 outColor;

vec3 worldPositionAtDepth(float depth)
{
    vec4 viewPosition = cameraUbo.invProj * vec4(inUV * 2.0 - 1.0, depth, 1.0);
    viewPosition /= max(abs(viewPosition.w), 1e-6);
    return (cameraUbo.invView * viewPosition).xyz;
}

float sphereHit(vec3 origin, vec3 direction, vec3 centre, float radius)
{
    vec3 offset = origin - centre;
    float b = dot(offset, direction);
    float determinant = b * b - (dot(offset, offset) - radius * radius);
    if (determinant < 0.0)
        return -1.0;
    float nearHit = -b - sqrt(determinant);
    float farHit = -b + sqrt(determinant);
    return nearHit > 0.0 ? nearHit : farHit;
}

void main()
{
    vec3 origin = cameraUbo.cameraPosition.xyz;
    vec3 direction = normalize(worldPositionAtDepth(1.0) - origin);

    float nearest = 1e30;
    vec3 hitColor = vec3(0.0);

    // A floor establishes the volume's base and makes its depth clipping easy to inspect.
    if (direction.y < -1e-5)
    {
        float floorDistance = -origin.y / direction.y;
        if (floorDistance > 0.0)
        {
            vec3 position = origin + direction * floorDistance;
            float grid = abs(fract(position.x * 2.0) - 0.5) < 0.035 ||
                         abs(fract(position.z * 2.0) - 0.5) < 0.035 ? 1.0 : 0.0;
            nearest = floorDistance;
            hitColor = mix(vec3(0.105, 0.12, 0.14), vec3(0.20, 0.23, 0.26), grid);
        }
    }

    // This matte sphere sits partly inside the volume. Smoke behind it must disappear at its depth,
    // while smoke between it and the camera remains visible.
    vec3 sphereCentre = vec3(0.38, 0.56, 0.28);
    float sphereDistance = sphereHit(origin, direction, sphereCentre, 0.42);
    if (sphereDistance > 0.0 && sphereDistance < nearest)
    {
        vec3 position = origin + direction * sphereDistance;
        vec3 normal = normalize(position - sphereCentre);
        float diffuse = 0.25 + 0.75 * max(dot(normal, normalize(vec3(0.5, 1.0, 0.3))), 0.0);
        nearest = sphereDistance;
        hitColor = vec3(0.22, 0.31, 0.38) * diffuse;
    }

    if (nearest < 1e29)
    {
        vec3 position = origin + direction * nearest;
        vec4 clip = cameraUbo.viewProj * vec4(position, 1.0);
        gl_FragDepth = clamp(clip.z / clip.w, 0.0, 1.0);
        outColor = vec4(hitColor, 1.0);
    }
    else
    {
        float horizon = clamp(direction.y * 0.5 + 0.5, 0.0, 1.0);
        outColor = vec4(mix(vec3(0.055, 0.07, 0.09), vec3(0.18, 0.23, 0.29), horizon), 1.0);
        gl_FragDepth = 1.0;
    }
}
