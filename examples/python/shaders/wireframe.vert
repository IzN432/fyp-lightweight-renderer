#version 450

layout(location = 0) in vec3 inPosition;

void main()
{
    // The example normalizes model coordinates to roughly [-0.9, 0.9]. Vulkan's
    // normalized depth range is [0, 1], so remap z while leaving x/y unchanged.
    gl_Position = vec4(inPosition.xy, inPosition.z * 0.5 + 0.5, 1.0);
}
