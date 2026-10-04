# Engine building blocks for Python renderers

The `lr` module lets Python build its own renderer on the engine's frame graph. These building blocks
are the C++ renderer's own pieces, usable from such a renderer:

- **`lr.SceneGpu`** uploads a scene in the engine's GPU layout and keeps it in sync.
- **The engine's passes** read and write named frame-graph resources. Your own passes can sit between
  them, reading their outputs and producing inputs for the next block.

```python
scene = lr.load_scene("model.glb")
scene.add_light("directional", rotation=(-0.383, 0, 0, 0.924))
scene.add_light("image")                        # ambient light from the Ibl

viewer = lr.Viewer()
camera = lr.OrbitCamera(viewer)
ibl = lr.Ibl(viewer, hdri="sky.hdr")            # precomputed now
gpu = lr.SceneGpu(viewer, scene, camera)        # scene -> GPU buffers, kept in sync
lr.GeometryPass(viewer, gpu)                    # -> G-buffer
lr.AmbientOcclusionPass(viewer, gpu)            # gbufferDepth -> hbao_ao
lr.PbrPass(viewer, gpu, ibl)                    # G-buffer + lights + IBL + AO -> pbr
# ... your passes: read "pbr", "gbufferDepth", ...; write e.g. "fogged"
lr.CompositePass(viewer, gpu, input="fogged")   # HDR -> window
viewer.run()
```

A complete example is [`examples/python/deferred_blocks.py`](../examples/python/deferred_blocks.py). It
inserts a fog pass, written in Python, between `PbrPass` and `CompositePass`.

## How blocks connect

- **Constructing a block declares its passes** in `viewer.frame_graph`. The Viewer holds the block
  until `run()` returns, so you don't need to keep a reference. Pass order follows from the resources
  passes use, as with your own passes.
- **Resources are connected by name.** A block reads images and buffers by their frame-graph names
  and writes its results under fixed names, listed below. To replace a block, write the names its
  consumers read. For example, any pass writing an R32F image called `hbao_ao` can stand in for
  `AmbientOcclusionPass`.
- **Every block reports its contract.** `block.inputs` and `block.outputs` are lists of `lr.ResourceUse`
  (`name`, `kind`, `usage`, `format`), and `block.describe()` prints them. They come from the passes'
  actual declarations, not from this document, so they are always current.
  - **Inputs** are what something else must provide.
  - **Outputs** are what the block makes for others.
  - A block's private data (for example HBAO's parameter buffer or PBR's LTC tables) and the
    intermediates between its own passes appear in neither list.
- **`block.passes`** holds the frame-graph pass handles, for `depends_on()`.

Attachment images (the G-buffer, `pbr`, your own outputs) are created by the frame graph at window
size, and resized with the window.

## `lr.SceneGpu(viewer, scene, camera)`

**What it does:** uploads every mesh and light in `scene`, and keeps the buffers current each frame:
- the camera, from `camera` (an `lr.OrbitCamera`);
- animation playback and GPU skinning palettes;
- light and material edits;
- lights added or removed while running.

**Changing lights while running:** each change shows from the next frame, and `GeometryPass` and
`PbrPass` follow automatically. [`examples/python/light_editor.py`](../examples/python/light_editor.py)
demonstrates all of these.
- `scene.add_light(...)` adds a light.
- `scene.remove(obj)` removes one.
- `obj.set_light(...)` changes its type, colour, intensity, area size, two-sidedness or cone angles.
- `obj.position` and `obj.rotation` move and aim it.

**Rules and side effects:**
- Use one per Viewer.
- At most `gpu.max_lights` (16) lights; more raises an error on the next frame.
- Every light gets a quad mesh drawn with the scene. Area lights show as emissive quads; other light
  types have invisible quads.
- The quads belong to the `SceneGpu`, not to the scene: the scene isn't modified, so it can be
  uploaded again later, for example by the next Viewer.

| Buffer (`gpu.buffers` key) | Name | Contents |
|---|---|---|
| `camera` | `camera_cb` | Uniform, std140, 336 bytes: `mat4 view, proj, viewProj, invView, invProj; vec3 position; float pad`. Vulkan clip space: depth in [0, 1], Y flipped. |
| `lights` | `lights_lb` | Storage buffer, one 80-byte entry per light: `vec3 position; uint type; vec4 rotation (quaternion xyzw); vec3 color; float intensity; float innerCone, outerCone (radians); vec2 areaSize; uint flags` (bit 0: two-sided area light), padded to 80. The type is 0 point, 1 spot, 2 area, 3 directional, 4 image. `gpu.num_lights` gives the count. |
| `positions` | `meshPositionBuffer` | Vertex buffer, binding 0: `vec3` position per render vertex, all meshes back to back. |
| `attributes` | `meshVertexBuffer` | Vertex buffer, binding 1: interleaved `vec3 normal, vec4 tangent (w = handedness), vec2 uv`. |
| `indices` | `meshIndexBuffer` | `uint32` triangle indices. |
| `face_groups` | `meshFaceGroupBuffer` | Storage buffer: one `uint` material slot per triangle, all meshes back to back. |
| `materials` | `material_info` | Storage buffer, 48-byte entries: `vec4 baseColorFactor; vec4 emissiveFactor; float roughness, metallic; float pad[2]`. One entry per material slot (256). |
| `joint_matrices` | `skinJointMatrices` | Skinning palettes (with `skinInfluenceEntries`, `skinInfluenceOffsets` and `skinPositionIndices`). |

**Material textures:** each texture is an image array with one element per material slot:
- `material_tex_baseColorTexture` (sRGB)
- `material_tex_normalTexture`
- `material_tex_metallicRoughnessTexture` (G = roughness, B = metallic)
- `material_tex_emissiveTexture` (sRGB)

## `lr.Ibl(viewer, hdri=None, env_res=2048, irr_res=32, pf_res=2048, pf_mips=8)`

Image-based lighting, precomputed once, when the block is constructed, in a separate frame graph.
`hdri` is an equirectangular `.hdr` image; with `None`, the environment is black.

| | Name | Format | Meaning |
|---|---|---|---|
| out | `ibl_env` | RGBA16F cubemap, `env_res`² | The environment itself (the sky) |
| out | `ibl_irradiance` | RGBA16F cubemap, `irr_res`² | Diffuse irradiance |
| out | `ibl_prefiltered` | RGBA16F cubemap, `pf_res`², `pf_mips` mips | Specular radiance, one mip per roughness step |
| out | `ibl_brdf_lut` | RGBA8 2D | Split-sum BRDF lookup table |

## `lr.GeometryPass(viewer, scene_gpu)`

Draws every `SceneGpu` mesh with its material. For each pixel:
- **albedo** = `baseColorTexture` × `baseDiffuse`;
- **normal** comes from the normal map, through the tangent frame;
- **roughness and metallic** = the `metallicRoughnessTexture` G and B channels × the material factors.

Setting `.skinning = False` draws skinned meshes in their rest pose.

| | Name | Format | Meaning |
|---|---|---|---|
| in | `SceneGpu` buffers and textures | | see above |
| out | `gbufferAlbedo` | RGBA16F | rgb = albedo (linear) |
| out | `gbufferNormal` | RG16F | View-space normal xy; z = √(1 − x² − y²), which is ≥ 0 because surfaces face the camera |
| out | `gbufferMaterial` | RGBA16 UNORM | r = roughness, g = metallic |
| out | `gbufferEmissive` | RGBA16F | rgb = emissive radiance |
| out | `gbufferDepth` | D32 | Hardware depth, cleared to 1.0 where nothing was drawn. View position = `invProj * vec4(uv * 2 - 1, depth, 1)`, divided by w |

## `lr.AmbientOcclusionPass(viewer, scene_gpu, sphere_radius=0.5, num_steps=16, num_dirs=8, tan_angle_bias=0.364, ao_scalar=2.0)`

Horizon-based ambient occlusion (HBAO) on deinterleaved quarter-resolution depth, followed by a
depth-aware blur.

**Parameters:** all can be changed while running, as properties of the same names.
- **`sphere_radius`:** the world-space sampling radius. Scale it to the scene; the C++ renderer uses
  2% of the model's size.
- **`num_steps` and `num_dirs`:** steps per direction, and the number of directions.
- **`tan_angle_bias`:** horizons below this slope are ignored.
- **`ao_scalar`:** the strength.

| | Name | Format | Meaning |
|---|---|---|---|
| in | `gbufferDepth` | D32 | from GeometryPass |
| in | `camera_cb` | | from SceneGpu |
| out | `hbao_ao` | R32F | Occlusion: 0 = open, 1 = fully occluded |

## `lr.PbrPass(viewer, scene_gpu, ibl)`

Deferred lighting with a Cook-Torrance BRDF (GGX, Smith, Schlick):
- **point, spot and directional lights** are evaluated as delta lights;
- **area lights** use linearly transformed cosines (LTC). They emit along their forward axis (local −Z),
  or from both faces if two-sided (the default);
- **`image` lights** apply the `Ibl` (diffuse irradiance plus split-sum specular), scaled by the light's
  colour × intensity and darkened by `hbao_ao`.

Emissive is added on top. Background pixels (depth 1) are black.

**Light count:** kept current by the `SceneGpu`, as lights are added or removed.

| | Name | Format | Meaning |
|---|---|---|---|
| in | `gbufferAlbedo`, `gbufferNormal`, `gbufferMaterial`, `gbufferEmissive`, `gbufferDepth` | | from GeometryPass |
| in | `hbao_ao` | R32F | from AmbientOcclusionPass, or your own |
| in | `ibl_irradiance`, `ibl_prefiltered`, `ibl_brdf_lut` | | from Ibl |
| in | `camera_cb`, `lights_lb` | | from SceneGpu |
| out | `pbr` | RGBA16F | Linear HDR radiance |

## `lr.CompositePass(viewer, scene_gpu, input="pbr", output="swapchain", output_format=None, name="composite")`

The engine's final image, without the editor overlays:
- where `gbufferDepth` is 1, it shows the `ibl_env` sky;
- elsewhere, it shows `input`.

Both are Reinhard tone mapped (`c / (c + 1)`) and written to `output`, in the swapchain's format by
default. The C++ editor's `FinalPass` shares this code (`utility/tonemap.glslh`).

| | Name | Format | Meaning |
|---|---|---|---|
| in | `input` (default `pbr`) | any sampled colour | Linear HDR colour |
| in | `gbufferDepth` | D32 | Selects sky or scene |
| in | `ibl_env` | cubemap | The sky |
| in | `camera_cb` | | For sky directions |
| out | `output` (default `swapchain`) | `output_format` | Tone-mapped colour |

## Writing a pass between blocks

Read engine outputs like any other image, and write a new name for the next block:

```python
fg.add_pass("fog")
  .type(lr.PassType.FULLSCREEN)
  .vert_shader(pathlib.Path(lr.SHADER_DIR) / "fullscreen.vert.spv")   # the engine's fullscreen triangle
  .frag_shader("fog.frag")
  .uniform_buffer(0, fg.buffer(gpu.camera_buffer), lr.Stage.FRAGMENT)
  .sampled_image(1, fg.image("pbr"), lr.Stage.FRAGMENT)
  .sampled_depth(2, fg.image("gbufferDepth"), lr.Stage.FRAGMENT)
  .color_attachment(fg.image("fogged"), lr.Format.R16G16B16A16_SFLOAT)
  .execute(draw)
lr.CompositePass(viewer, gpu, input="fogged")
```

**Checks:**
- Shader interfaces are checked against these declarations when the graph compiles.
- If a block's input is missing, for example `PbrPass` without `hbao_ao`, compiling the graph names the
  missing resource.
