#version 450

// OutlinePass: Blender's selected-object outline, read out of the editor's picking ID buffer.
//
// Every pixel holds the ID of the object ObjectPickingPass drew there (0 = nothing), and the
// selection is a bitset indexed by that ID, so "is this pixel part of the selection" is one texel
// fetch and one bit test. The outline is the set of *unselected* pixels within `thickness` pixels of
// a selected one: like Blender's, it sits outside the silhouette rather than eating into the object,
// it follows holes and concavities for free, and it stops where a nearer object occludes the
// selection — the occluder's own ID is what landed in the buffer there.
//
// The distance to the nearest selected pixel also gives the edge its antialiasing: full opacity up
// to `thickness`, falling off over the last pixel.

layout(location = 0) in vec2 inUV;

layout(set = 0, binding = 0) uniform usampler2D pickingIds;

layout(std430, set = 0, binding = 1) readonly buffer SelectedObjects { uint words[]; } selection;

layout(push_constant) uniform OutlinePC
{
    vec4  color;     // rgb = outline colour, a = peak opacity
    float thickness; // outline half-width, in pixels of the picking buffer
    uint  wordCount; // words of `selection` the CPU filled in; IDs past it are unselected
} pc;

layout(location = 0) out vec4 outColor;

bool isSelected(ivec2 coord, ivec2 size)
{
    // Clamping rather than discarding off-screen taps keeps the outline continuous where the
    // selection runs off the edge of the viewport.
    ivec2 clamped = clamp(coord, ivec2(0), size - ivec2(1));
    uint  id      = texelFetch(pickingIds, clamped, 0).r;
    if (id == 0u)
    {
        return false;
    }
    uint word = id >> 5u;
    return word < pc.wordCount && (selection.words[word] & (1u << (id & 31u))) != 0u;
}

void main()
{
    ivec2 size   = textureSize(pickingIds, 0);
    ivec2 center = ivec2(inUV * vec2(size));
    if (isSelected(center, size))
    {
        discard;
    }

    int   radius  = int(ceil(pc.thickness));
    float nearest = 1e9;
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            if (isSelected(center + ivec2(x, y), size))
            {
                nearest = min(nearest, length(vec2(x, y)));
            }
        }
    }

    float coverage = clamp(pc.thickness + 0.5 - nearest, 0.0, 1.0);
    if (coverage <= 0.0)
    {
        discard;
    }
    outColor = vec4(pc.color.rgb, coverage * pc.color.a);
}
