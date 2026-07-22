#version 450

layout(location = 0) in vec3 inColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D gbufferDepth;

layout(push_constant) uniform PC {
    layout(offset = 64) float pointSize;
    layout(offset = 68) float occludedOpacity;
} pc;

void main()
{
    // Clip to circle shape
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25) discard;

    vec2 screenSize = vec2(textureSize(gbufferDepth, 0));

    // Sample depth at the sprite center so the whole point shares one occlusion decision.
    // gl_FragCoord.z is the same for all fragments in a point sprite, so we only need to
    // redirect the depth *sample* to the center pixel.
    vec2 centerFragCoord = gl_FragCoord.xy + (vec2(0.5) - gl_PointCoord) * pc.pointSize;
    float sceneDepth = texture(gbufferDepth, centerFragCoord / screenSize).r;

    bool isOccluded = gl_FragCoord.z > sceneDepth + 1e-4;
    outColor = vec4(inColor, isOccluded ? pc.occludedOpacity : 1.0);
}
