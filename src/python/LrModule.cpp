// Python bindings for the frame graph: Viewer, ResourceRegistry, FrameGraph, PassBuilder,
// PassContext and CommandBuffer, plus runtime GLSL compilation. Built as `lr._lr` and re-exported
// by the pure-Python `lr` package (python/lr/__init__.py).

#include "core/Paths.hpp"
#include "core/app/Viewer.hpp"
#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/PassBuilder.hpp"
#include "core/framegraph/PassContext.hpp"
#include "core/framegraph/ResourceRegistry.hpp"
#include "core/framegraph/compiler/ShaderInterface.hpp"
#include "core/vulkan/CommandBuffer.hpp"
#include "core/vulkan/ShaderCompiler.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#if defined(_MSC_VER) && defined(_DEBUG)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <crtdbg.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <unordered_map>

namespace nb = nanobind;
using namespace nb::literals;
namespace fs = std::filesystem;

namespace
{

using lr::BufferHandle;
using lr::CommandBuffer;
using lr::ExtentSpec;
using lr::FrameGraph;
using lr::ImageHandle;
using lr::PassBuilder;
using lr::PassContext;
using lr::PassHandle;
using lr::PassType;
using lr::ResourceRegistry;
using lr::ShaderStage;
using lr::Viewer;

// Any C-contiguous host array (numpy array, bytes, bytearray, memoryview...). Non-contiguous numpy
// arrays are copied to a contiguous one by nanobind before the call.
using HostArray = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

// Python callbacks run inside the C++ frame loop, so an exception unwinding through it would abandon
// a half-recorded command buffer. Instead the first one is stashed here, the window is asked to close
// so run() returns after the current frame, and run() re-raises it (with its traceback).
// Raised from run() for validation-layer errors (Viewer(raise_validation_errors=True), the default).
class VulkanValidationError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct CallbackErrorTrap
{
    Viewer                         *running = nullptr;
    std::optional<nb::python_error> error;

    // Validation errors arrive through a C++ handler, outside any Python call, so they're kept as
    // text with the Viewer that produced them; run() raises the first one that belongs to it.
    Viewer                    *validationSource = nullptr;
    std::optional<std::string> validationError;
    size_t                     laterValidationErrors = 0;

    void recordValidationError(Viewer &viewer, std::string_view message)
    {
        if (validationError && validationSource == &viewer)
        {
            ++laterValidationErrors;
            return;
        }
        validationSource      = &viewer;
        validationError       = std::string(message);
        laterValidationErrors = 0;
        if (running == &viewer)
        {
            viewer.requestClose();
        }
    }

    void clearValidationErrors()
    {
        validationSource = nullptr;
        validationError.reset();
        laterValidationErrors = 0;
    }

    template <typename F> void guard(F &&callback)
    {
        if (!running)
        {
            callback();
            return;
        }
        try
        {
            callback();
        } catch (nb::python_error &e)
        {
            if (!error)
            {
                error.emplace(std::move(e));
            }
            running->requestClose();
        }
    }

    // A Python exception from a callback takes priority: it's usually the cause of what followed.
    void rethrowPending(Viewer &viewer)
    {
        const bool        hasValidation = validationError && validationSource == &viewer;
        const std::string validation    = hasValidation ? *validationError : std::string();
        const size_t      later         = laterValidationErrors;
        clearValidationErrors();

        if (error)
        {
            nb::python_error pending = std::move(*error);
            error.reset();
            throw pending;
        }
        if (hasValidation)
        {
            std::string message =
                "Vulkan validation error (the window closed after the frame that caused it):\n" + validation;
            if (later > 0)
            {
                message += "\n(" + std::to_string(later) + " more validation error(s) followed; see the log)";
            }
            // The layer can't tell a mistake in the script from one in lr's own Vulkan usage.
            message += "\nIf your passes and draw calls look correct, this may be a bug in lr itself: please report "
                       "it with this message and the log.";
            throw VulkanValidationError(message);
        }
    }
};

CallbackErrorTrap g_callbackErrors;

// Python callables handed to C++ (frame callbacks, pass execute callbacks) usually close over the
// Viewer itself — a reference cycle through C++ that Python's GC can't see, which would keep the
// Viewer (and its Vulkan device) alive until process exit. The C++ closures own their callable
// through a shared slot; this registry only tracks the slots weakly, so it never outlives the
// interpreter itself, and releaseAll() breaks every cycle at once.
struct CallbackSlots
{
    std::vector<std::weak_ptr<nb::object>> slots;

    std::shared_ptr<nb::object> hold(nb::callable callback)
    {
        auto slot = std::make_shared<nb::object>(std::move(callback));
        slots.push_back(slot);
        return slot;
    }

    void releaseAll()
    {
        for (const std::weak_ptr<nb::object> &weak : slots)
        {
            if (const std::shared_ptr<nb::object> slot = weak.lock())
            {
                *slot = nb::object();
            }
        }
        slots.clear();
    }
};

CallbackSlots g_callbackSlots;

nb::tuple toTuple(VkExtent2D extent) { return nb::make_tuple(extent.width, extent.height); }

nb::bytes spirvToBytes(const std::vector<uint32_t> &spirv)
{
    return nb::bytes(reinterpret_cast<const char *>(spirv.data()), spirv.size() * sizeof(uint32_t));
}

std::vector<uint32_t> bytesToSpirv(const nb::bytes &bytes)
{
    if (bytes.size() % sizeof(uint32_t) != 0)
    {
        throw std::invalid_argument("SPIR-V byte length must be a multiple of 4");
    }
    std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
    std::memcpy(words.data(), bytes.c_str(), bytes.size());
    return words;
}

bool isSpirvFile(const fs::path &path) { return path.extension() == ".spv"; }

// Bytes per texel for the formats upload_image accepts.
uint32_t texelSize(VkFormat format)
{
    static const std::unordered_map<VkFormat, uint32_t> sizes = {
        {VK_FORMAT_R8_UNORM, 1},
        {VK_FORMAT_R8G8B8A8_UNORM, 4},
        {VK_FORMAT_R8G8B8A8_SRGB, 4},
        {VK_FORMAT_B8G8R8A8_UNORM, 4},
        {VK_FORMAT_B8G8R8A8_SRGB, 4},
        {VK_FORMAT_R16_SFLOAT, 2},
        {VK_FORMAT_R16G16B16A16_SFLOAT, 8},
        {VK_FORMAT_R32_SFLOAT, 4},
        {VK_FORMAT_R32G32_SFLOAT, 8},
        {VK_FORMAT_R32G32B32A32_SFLOAT, 16},
        {VK_FORMAT_R32_UINT, 4},
    };
    const auto it = sizes.find(format);
    if (it == sizes.end())
    {
        throw std::invalid_argument("upload_image: format not supported for uploads");
    }
    return it->second;
}

// (width, height) of an array shaped (height, width[, channels]) holding texels of `format`.
std::pair<uint32_t, uint32_t> imageSize(const HostArray &data, VkFormat format, const char *function)
{
    if (data.ndim() < 2)
    {
        throw std::invalid_argument(std::string(function) + ": expected an array shaped (height, width[, channels])");
    }
    const auto height = static_cast<uint32_t>(data.shape(0));
    const auto width  = static_cast<uint32_t>(data.shape(1));
    if (data.nbytes() != static_cast<size_t>(width) * height * texelSize(format))
    {
        throw std::invalid_argument(std::string(function) +
                                    ": array byte size doesn't match height * width * bytes-per-texel of the format");
    }
    return {width, height};
}

VkClearValue colorClear(const std::array<float, 4> &rgba)
{
    VkClearValue value{};
    std::copy(rgba.begin(), rgba.end(), value.color.float32);
    return value;
}

void bindEnums(nb::module_ &m)
{
    nb::enum_<VkFormat>(m, "Format")
        .value("UNDEFINED", VK_FORMAT_UNDEFINED)
        .value("R8_UNORM", VK_FORMAT_R8_UNORM)
        .value("R8G8B8A8_UNORM", VK_FORMAT_R8G8B8A8_UNORM)
        .value("R8G8B8A8_SRGB", VK_FORMAT_R8G8B8A8_SRGB)
        .value("B8G8R8A8_UNORM", VK_FORMAT_B8G8R8A8_UNORM)
        .value("B8G8R8A8_SRGB", VK_FORMAT_B8G8R8A8_SRGB)
        .value("R16_SFLOAT", VK_FORMAT_R16_SFLOAT)
        .value("R16G16B16A16_SFLOAT", VK_FORMAT_R16G16B16A16_SFLOAT)
        .value("R32_SFLOAT", VK_FORMAT_R32_SFLOAT)
        .value("R32G32_SFLOAT", VK_FORMAT_R32G32_SFLOAT)
        .value("R32G32B32_SFLOAT", VK_FORMAT_R32G32B32_SFLOAT)
        .value("R32G32B32A32_SFLOAT", VK_FORMAT_R32G32B32A32_SFLOAT)
        .value("R32_UINT", VK_FORMAT_R32_UINT)
        .value("D32_SFLOAT", VK_FORMAT_D32_SFLOAT);

    nb::enum_<VkBufferUsageFlagBits>(m, "BufferUsage", nb::is_flag(), nb::is_arithmetic())
        .value("TRANSFER_SRC", VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        .value("TRANSFER_DST", VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        .value("UNIFORM", VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
        .value("STORAGE", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        .value("INDEX", VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
        .value("VERTEX", VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
        .value("INDIRECT", VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

    nb::enum_<VkImageUsageFlagBits>(m, "ImageUsage", nb::is_flag(), nb::is_arithmetic())
        .value("TRANSFER_SRC", VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        .value("TRANSFER_DST", VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .value("SAMPLED", VK_IMAGE_USAGE_SAMPLED_BIT)
        .value("STORAGE", VK_IMAGE_USAGE_STORAGE_BIT)
        .value("COLOR_ATTACHMENT", VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        .value("DEPTH_STENCIL_ATTACHMENT", VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

    nb::enum_<VkShaderStageFlagBits>(m, "Stage", nb::is_flag(), nb::is_arithmetic())
        .value("VERTEX", VK_SHADER_STAGE_VERTEX_BIT)
        .value("FRAGMENT", VK_SHADER_STAGE_FRAGMENT_BIT)
        .value("COMPUTE", VK_SHADER_STAGE_COMPUTE_BIT)
        .value("ALL_GRAPHICS", VK_SHADER_STAGE_ALL_GRAPHICS);

    nb::enum_<VkPrimitiveTopology>(m, "Topology")
        .value("POINT_LIST", VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
        .value("LINE_LIST", VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
        .value("LINE_STRIP", VK_PRIMITIVE_TOPOLOGY_LINE_STRIP)
        .value("TRIANGLE_LIST", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .value("TRIANGLE_STRIP", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);

    nb::enum_<VkAttachmentLoadOp>(m, "LoadOp")
        .value("LOAD", VK_ATTACHMENT_LOAD_OP_LOAD)
        .value("CLEAR", VK_ATTACHMENT_LOAD_OP_CLEAR)
        .value("DONT_CARE", VK_ATTACHMENT_LOAD_OP_DONT_CARE);

    nb::enum_<lr::BlendMode>(m, "BlendMode")
        .value("OPAQUE", lr::BlendMode::Opaque)
        .value("ALPHA", lr::BlendMode::Alpha)
        .value("PREMULTIPLIED_ALPHA", lr::BlendMode::PremultipliedAlpha)
        .value("ADDITIVE", lr::BlendMode::Additive);

    nb::enum_<VkPolygonMode>(m, "PolygonMode")
        .value("FILL", VK_POLYGON_MODE_FILL)
        .value("LINE", VK_POLYGON_MODE_LINE)
        .value("POINT", VK_POLYGON_MODE_POINT);

    nb::enum_<VkCullModeFlagBits>(m, "CullMode", nb::is_flag(), nb::is_arithmetic())
        .value("NONE", VK_CULL_MODE_NONE)
        .value("FRONT", VK_CULL_MODE_FRONT_BIT)
        .value("BACK", VK_CULL_MODE_BACK_BIT)
        .value("FRONT_AND_BACK", VK_CULL_MODE_FRONT_AND_BACK);

    nb::enum_<VkFrontFace>(m, "FrontFace")
        .value("COUNTER_CLOCKWISE", VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .value("CLOCKWISE", VK_FRONT_FACE_CLOCKWISE);

    nb::enum_<VkCompareOp>(m, "CompareOp")
        .value("NEVER", VK_COMPARE_OP_NEVER)
        .value("LESS", VK_COMPARE_OP_LESS)
        .value("EQUAL", VK_COMPARE_OP_EQUAL)
        .value("LESS_OR_EQUAL", VK_COMPARE_OP_LESS_OR_EQUAL)
        .value("GREATER", VK_COMPARE_OP_GREATER)
        .value("NOT_EQUAL", VK_COMPARE_OP_NOT_EQUAL)
        .value("GREATER_OR_EQUAL", VK_COMPARE_OP_GREATER_OR_EQUAL)
        .value("ALWAYS", VK_COMPARE_OP_ALWAYS);

    nb::enum_<PassType>(m, "PassType")
        .value("GEOMETRY", PassType::Geometry)
        .value("FULLSCREEN", PassType::Fullscreen)
        .value("COMPUTE", PassType::Compute)
        .value("CUSTOM", PassType::Custom);

    nb::enum_<ShaderStage>(m, "ShaderStage")
        .value("VERTEX", ShaderStage::Vertex)
        .value("FRAGMENT", ShaderStage::Fragment)
        .value("COMPUTE", ShaderStage::Compute);
}

void bindValueTypes(nb::module_ &m)
{
    nb::class_<ImageHandle>(m, "ImageHandle")
        .def("__bool__",
             [](ImageHandle h) {
                 return static_cast<bool>(h);
             })
        .def("__eq__", [](ImageHandle a, ImageHandle b) {
            return a == b;
        });
    nb::class_<BufferHandle>(m, "BufferHandle")
        .def("__bool__",
             [](BufferHandle h) {
                 return static_cast<bool>(h);
             })
        .def("__eq__", [](BufferHandle a, BufferHandle b) {
            return a == b;
        });
    nb::class_<PassHandle>(m, "PassHandle")
        .def("__bool__",
             [](PassHandle h) {
                 return static_cast<bool>(h);
             })
        .def("__eq__", [](PassHandle a, PassHandle b) {
            return a == b;
        });

    nb::class_<ExtentSpec>(m, "Extent", "How big a frame-graph image is: fixed, or tracking the swapchain.")
        .def_static("swapchain", &ExtentSpec::swapchain, "Same size as the swapchain (the default).")
        .def_static("absolute", &ExtentSpec::absolute, "width"_a, "height"_a, "A fixed size in pixels.")
        .def_static(
            "relative",
            [](uint32_t numerator, uint32_t denominator) {
                return ExtentSpec::relative(numerator, denominator);
            },
            "numerator"_a, "denominator"_a, "A fraction of the swapchain size, e.g. relative(1, 2) for half-res.");

    nb::class_<VkVertexInputBindingDescription>(m, "VertexBinding")
        .def(
            "__init__",
            [](VkVertexInputBindingDescription *self, uint32_t binding, uint32_t stride, bool perInstance) {
                new (self) VkVertexInputBindingDescription{
                    .binding   = binding,
                    .stride    = stride,
                    .inputRate = perInstance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
                };
            },
            "binding"_a, "stride"_a, "per_instance"_a = false)
        .def_rw("binding", &VkVertexInputBindingDescription::binding)
        .def_rw("stride", &VkVertexInputBindingDescription::stride);

    nb::class_<VkVertexInputAttributeDescription>(m, "VertexAttribute")
        .def(
            "__init__",
            [](VkVertexInputAttributeDescription *self, uint32_t location, VkFormat format, uint32_t offset,
               uint32_t binding) {
                new (self) VkVertexInputAttributeDescription{
                    .location = location,
                    .binding  = binding,
                    .format   = format,
                    .offset   = offset,
                };
            },
            "location"_a, "format"_a, "offset"_a = 0, "binding"_a = 0)
        .def_rw("location", &VkVertexInputAttributeDescription::location)
        .def_rw("binding", &VkVertexInputAttributeDescription::binding)
        .def_rw("format", &VkVertexInputAttributeDescription::format)
        .def_rw("offset", &VkVertexInputAttributeDescription::offset);
}

void bindResources(nb::module_ &m)
{
    nb::class_<ResourceRegistry>(m, "ResourceRegistry",
                                 "Named GPU buffers and images. Passes refer to them by name via "
                                 "FrameGraph.buffer()/image().")
        .def(
            "upload_buffer",
            [](ResourceRegistry &r, const std::string &name, HostArray data, VkBufferUsageFlags usage) {
                r.uploadBuffer(name, data.data(), data.nbytes(), usage);
            },
            "name"_a, "data"_a, "usage"_a,
            "Create a GPU-only buffer sized to `data` and upload it before the next frame.")
        .def(
            "reupload_buffer",
            [](ResourceRegistry &r, const std::string &name, HostArray data) {
                r.reuploadBuffer(name, data.data(), data.nbytes());
            },
            "name"_a, "data"_a, "Overwrite an uploaded buffer; `data` must fit its original size.")
        .def("register_static_buffer", &ResourceRegistry::registerStaticBuffer, "name"_a, "size"_a, "usage"_a,
             "Create an uninitialised GPU-only buffer (e.g. compute scratch).")
        .def("register_dynamic_buffer", &ResourceRegistry::registerDynamicBuffer, "name"_a, "size"_a, "usage"_a,
             "Create a CPU-writable buffer for per-frame data such as uniforms; fill it with update_buffer().")
        .def(
            "update_buffer",
            [](ResourceRegistry &r, const std::string &name, HostArray data) {
                r.updateBuffer(name, data.data(), data.nbytes());
            },
            "name"_a, "data"_a, "Write `data` into a dynamic buffer (call from an on_update callback).")
        .def(
            "register_image",
            [](ResourceRegistry &r, const std::string &name, VkFormat format, VkImageUsageFlags usage,
               ExtentSpec extent) {
                r.registerImage(name, format, usage, extent);
            },
            "name"_a, "format"_a, "usage"_a, "extent"_a = ExtentSpec::swapchain(),
            "Create a transient image, reallocated when the window resizes. Images used as attachments "
            "are created automatically; register them only to choose usage or extent up front.")
        .def(
            "upload_image",
            [](ResourceRegistry &r, const std::string &name, HostArray data, VkFormat format, bool generateMipmaps) {
                const auto [width, height] = imageSize(data, format, "upload_image");
                r.uploadImage(name, data.data(), width, height, format, generateMipmaps);
            },
            "name"_a, "data"_a, "format"_a, "generate_mipmaps"_a = false,
            "Create a sampled image from an array shaped (height, width[, channels]).")
        .def(
            "replace_image",
            [](ResourceRegistry &r, const std::string &name, HostArray data, VkFormat format, bool generateMipmaps) {
                const auto [width, height] = imageSize(data, format, "replace_image");
                r.replaceUploadedImage(name, data.data(), width, height, format, generateMipmaps);
            },
            "name"_a, "data"_a, "format"_a, "generate_mipmaps"_a = false,
            "Replace an image created by upload_image() (any size/format). Safe while running: the old image "
            "is kept until in-flight frames finish, and passes sampling it recompile on the next frame.")
        .def(
            "replace_buffer",
            [](ResourceRegistry &r, const std::string &name, HostArray data, VkBufferUsageFlags usage) {
                r.replaceUploadedBuffer(name, data.data(), data.nbytes(), usage);
            },
            "name"_a, "data"_a, "usage"_a,
            "Replace a buffer created by upload_buffer() with new contents of any size. Safe while running: the "
            "old buffer is kept until in-flight frames finish. Vertex/index/indirect buffers take effect "
            "immediately; passes binding it as a uniform/storage buffer recompile on the next frame.")
        .def("replace_dynamic_buffer", &ResourceRegistry::replaceDynamicBuffer, "name"_a, "size"_a, "usage"_a,
             "Reallocate a buffer created by register_dynamic_buffer() at a new size (contents start undefined). "
             "Same safety as replace_buffer().")
        .def(
            "read_buffer",
            [](ResourceRegistry &r, const std::string &name) {
                auto       *bytes = new std::vector<std::byte>(r.readBuffer(name));
                nb::capsule owner(bytes, [](void *p) noexcept {
                    delete static_cast<std::vector<std::byte> *>(p);
                });
                return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(bytes->data(), {bytes->size()}, owner);
            },
            "name"_a,
            "Copy a buffer back to the CPU as a uint8 numpy array (use .view(np.float32) etc.), as of the last "
            "submitted frame. Waits for the GPU to go idle: meant for tests and debugging, not every frame.")
        .def("has_buffer", &ResourceRegistry::hasBuffer, "name"_a)
        .def("has_image", &ResourceRegistry::hasImage, "name"_a)
        .def_prop_ro(
            "extent",
            [](const ResourceRegistry &r) {
                return toTuple(r.getExtent());
            },
            "Current swapchain-sized extent as (width, height).");
}

void bindPasses(nb::module_ &m)
{
    nb::class_<CommandBuffer>(m, "CommandBuffer")
        .def("draw", &CommandBuffer::draw, "vertex_count"_a, "instance_count"_a = 1, "first_vertex"_a = 0,
             "first_instance"_a = 0)
        .def("draw_indexed", &CommandBuffer::drawIndexed, "index_count"_a, "instance_count"_a = 1, "first_index"_a = 0,
             "vertex_offset"_a = 0, "first_instance"_a = 0)
        .def("dispatch", &CommandBuffer::dispatch, "x"_a, "y"_a = 1, "z"_a = 1)
        .def("set_viewport", &CommandBuffer::setViewport, "x"_a, "y"_a, "width"_a, "height"_a, "min_depth"_a = 0.0f,
             "max_depth"_a = 1.0f)
        .def("set_scissor", &CommandBuffer::setScissor, "x"_a, "y"_a, "width"_a, "height"_a);

    nb::class_<PassContext>(m, "PassContext",
                            "Handed to a pass's execute callback. Only valid during that call — don't keep it.")
        .def_prop_ro("cmd", &PassContext::cmd, nb::rv_policy::reference_internal)
        .def_prop_ro("rendering_extent",
                     [](const PassContext &ctx) {
                         return toTuple(ctx.renderingExtent());
                     })
        .def(
            "extent",
            [](const PassContext &ctx, ImageHandle image) {
                return toTuple(ctx.extent(image));
            },
            "image"_a)
        .def("draw_indirect", &PassContext::drawIndirect, "buffer"_a, "draw_count"_a = 1, "offset"_a = 0,
             "stride"_a = uint32_t(sizeof(VkDrawIndirectCommand)),
             "vkCmdDrawIndirect from a buffer declared with indirect_buffer(); commands are 4 uint32s "
             "(vertex_count, instance_count, first_vertex, first_instance).")
        .def("draw_indexed_indirect", &PassContext::drawIndexedIndirect, "buffer"_a, "draw_count"_a = 1, "offset"_a = 0,
             "stride"_a = uint32_t(sizeof(VkDrawIndexedIndirectCommand)),
             "vkCmdDrawIndexedIndirect; commands are 5 x 4 bytes (index_count, instance_count, first_index, "
             "vertex_offset, first_instance).")
        .def("dispatch_indirect", &PassContext::dispatchIndirect, "buffer"_a, "offset"_a = 0,
             "vkCmdDispatchIndirect; the command is 3 uint32s (x, y, z).")
        .def(
            "push_constants",
            [](PassContext &ctx, VkShaderStageFlags stages, HostArray data, uint32_t offset) {
                const size_t size = data.nbytes();
                if (size % 4 != 0 || offset % 4 != 0)
                {
                    throw std::invalid_argument("push_constants: offset and byte size must be multiples of 4");
                }
                if (offset + size > ctx.pushConstantSize())
                {
                    throw std::invalid_argument("push_constants: writes bytes [" + std::to_string(offset) + ", " +
                                                std::to_string(offset + size) +
                                                ") but the pass declares push_constant_size(" +
                                                std::to_string(ctx.pushConstantSize()) + ")");
                }
                if ((stages & ~ctx.pushConstantStages()) != 0)
                {
                    throw std::invalid_argument("push_constants: stages must be among those given to "
                                                "push_constant_size()");
                }
                ctx.cmd().pushConstants(ctx.pipelineLayout(), stages, data.data(), static_cast<uint32_t>(size), offset);
            },
            "stages"_a, "data"_a, "offset"_a = 0,
            "Push `data` (e.g. a float32 numpy array) into the pass's push-constant block.");

    const auto ref = nb::rv_policy::reference;

    nb::class_<PassBuilder>(m, "PassBuilder", "Declares one pass. Every method returns the builder for chaining.")
        .def_prop_ro("handle", &PassBuilder::handle)
        .def("type", &PassBuilder::type, "type"_a, ref)
        // Shaders: bytes are SPIR-V; a path ending in .spv is loaded as SPIR-V; any other path is GLSL,
        // compiled now (so syntax errors raise ShaderCompileError here, at the declaring line).
        .def(
            "vert_shader",
            [](PassBuilder &b, nb::bytes spirv) -> PassBuilder & {
                return b.vertShader(bytesToSpirv(spirv), "vertex shader (SPIR-V bytes)");
            },
            "spirv"_a, ref)
        .def(
            "vert_shader",
            [](PassBuilder &b, const fs::path &path) -> PassBuilder & {
                return isSpirvFile(path) ? b.vertShader(path.string())
                                         : b.vertShader(lr::compileGlslFile(path, ShaderStage::Vertex), path.string());
            },
            "path"_a, ref)
        .def(
            "frag_shader",
            [](PassBuilder &b, nb::bytes spirv) -> PassBuilder & {
                return b.fragShader(bytesToSpirv(spirv), "fragment shader (SPIR-V bytes)");
            },
            "spirv"_a, ref)
        .def(
            "frag_shader",
            [](PassBuilder &b, const fs::path &path) -> PassBuilder & {
                return isSpirvFile(path)
                           ? b.fragShader(path.string())
                           : b.fragShader(lr::compileGlslFile(path, ShaderStage::Fragment), path.string());
            },
            "path"_a, ref)
        .def(
            "compute_shader",
            [](PassBuilder &b, nb::bytes spirv) -> PassBuilder & {
                return b.computeShader(bytesToSpirv(spirv), "compute shader (SPIR-V bytes)");
            },
            "spirv"_a, ref)
        .def(
            "compute_shader",
            [](PassBuilder &b, const fs::path &path) -> PassBuilder & {
                return isSpirvFile(path)
                           ? b.computeShader(path.string())
                           : b.computeShader(lr::compileGlslFile(path, ShaderStage::Compute), path.string());
            },
            "path"_a, ref)
        .def("push_constant_size", &PassBuilder::pushConstantSize, "size"_a, "stages"_a, ref)
        .def("topology", &PassBuilder::topology, "topology"_a, ref)
        .def(
            "vertex_layout",
            [](PassBuilder &b, std::vector<VkVertexInputBindingDescription> bindings,
               std::vector<VkVertexInputAttributeDescription> attributes) -> PassBuilder & {
                return b.vertexLayout(std::move(bindings), std::move(attributes));
            },
            "bindings"_a, "attributes"_a, ref)
        .def("blend", &PassBuilder::blend, "mode"_a, ref,
             "Blend mode for all color attachments. Blending reads them, so this pass also runs after "
             "whatever wrote them earlier (use load_op=LOAD to blend over that content).")
        .def("polygon_mode", &PassBuilder::polygonMode, "mode"_a, ref, "FILL (default), LINE (wireframe) or POINT.")
        .def("cull", &PassBuilder::cull, "mode"_a, "front_face"_a = VK_FRONT_FACE_COUNTER_CLOCKWISE, ref,
             "Default: BACK for geometry passes, NONE for fullscreen ones.")
        .def("depth", &PassBuilder::depth, "test"_a, "write"_a, "compare"_a = VK_COMPARE_OP_LESS, ref,
             "Default: test and write on (compare LESS) exactly when the pass has a depth attachment.")
        .def("depth_bias", &PassBuilder::depthBias, "constant"_a, "slope"_a = 0.0f, ref,
             "Offset rasterized depth, e.g. negative values to draw a wireframe over its own solid surface.")
        .def("sampled_image", &PassBuilder::sampledImage, "binding"_a, "image"_a, "stages"_a, ref)
        .def("sampled_depth", &PassBuilder::sampledDepth, "binding"_a, "image"_a, "stages"_a, ref)
        .def(
            "storage_image_read",
            [](PassBuilder &b, uint32_t binding, ImageHandle image, VkShaderStageFlags stages) -> PassBuilder & {
                return b.storageImageRead(binding, image, stages);
            },
            "binding"_a, "image"_a, "stages"_a, ref)
        .def(
            "storage_image_write",
            [](PassBuilder &b, uint32_t binding, ImageHandle image, VkShaderStageFlags stages) -> PassBuilder & {
                return b.storageImageWrite(binding, image, stages);
            },
            "binding"_a, "image"_a, "stages"_a, ref)
        .def(
            "storage_image_read_write",
            [](PassBuilder &b, uint32_t binding, ImageHandle image, VkShaderStageFlags stages) -> PassBuilder & {
                return b.storageImageReadWrite(binding, image, stages);
            },
            "binding"_a, "image"_a, "stages"_a, ref)
        .def("uniform_buffer", &PassBuilder::uniformBuffer, "binding"_a, "buffer"_a, "stages"_a, ref)
        .def("storage_buffer_read", &PassBuilder::storageBufferRead, "binding"_a, "buffer"_a, "stages"_a, ref)
        .def("storage_buffer_write", &PassBuilder::storageBufferWrite, "binding"_a, "buffer"_a, "stages"_a, ref)
        .def("storage_buffer_read_write", &PassBuilder::storageBufferReadWrite, "binding"_a, "buffer"_a, "stages"_a,
             ref)
        .def("vertex_buffer", &PassBuilder::vertexBuffer, "binding"_a, "buffer"_a, ref)
        .def("index_buffer", &PassBuilder::indexBuffer, "buffer"_a, ref)
        .def("indirect_buffer", &PassBuilder::indirectBuffer, "buffer"_a, ref,
             "Declare a buffer of draw/dispatch arguments for PassContext.draw_indirect() and friends, e.g. "
             "written by a compute pass; the frame graph inserts the barrier between them.")
        .def("runs_last", &PassBuilder::runsLast, ref,
             "Order this pass after every other pass sharing a resource with it, even ones declared later.")
        .def(
            "color_attachment",
            [](PassBuilder &b, ImageHandle image, VkFormat format, VkAttachmentLoadOp loadOp,
               std::array<float, 4> clearColor, ExtentSpec extent) -> PassBuilder & {
                return b.colorAttachment(image, format, loadOp, colorClear(clearColor), extent);
            },
            "image"_a, "format"_a, "load_op"_a = VK_ATTACHMENT_LOAD_OP_CLEAR,
            "clear_color"_a = std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}, "extent"_a = ExtentSpec::swapchain(), ref)
        .def(
            "depth_attachment",
            [](PassBuilder &b, ImageHandle image, VkFormat format, VkAttachmentLoadOp loadOp, float clearDepth,
               ExtentSpec extent) -> PassBuilder & {
                VkClearValue clear{};
                clear.depthStencil = {clearDepth, 0};
                return b.depthAttachment(image, format, loadOp, clear, extent);
            },
            "image"_a, "format"_a = VK_FORMAT_D32_SFLOAT, "load_op"_a = VK_ATTACHMENT_LOAD_OP_CLEAR,
            "clear_depth"_a = 1.0f, "extent"_a = ExtentSpec::swapchain(), ref)
        .def(
            "depends_on",
            [](PassBuilder &b, PassHandle dep) -> PassBuilder & {
                return b.dependsOn(dep);
            },
            "dependency"_a, ref)
        .def(
            "depends_on",
            [](PassBuilder &b, std::vector<PassHandle> deps) -> PassBuilder & {
                return b.dependsOn(std::move(deps));
            },
            "dependencies"_a, ref)
        .def(
            "execute",
            [](PassBuilder &b, nb::callable callback) -> PassBuilder & {
                return b.execute([slot = g_callbackSlots.hold(std::move(callback))](PassContext &ctx) {
                    if (*slot)
                    {
                        g_callbackErrors.guard([&] {
                            (*slot)(nb::cast(&ctx, nb::rv_policy::reference));
                        });
                    }
                });
            },
            "callback"_a, "Record this pass's commands: callback(ctx: PassContext), called every frame.", ref);

    nb::class_<FrameGraph>(m, "FrameGraph")
        .def("add_pass", &FrameGraph::addPass, "name"_a, nb::keep_alive<0, 1>(),
             "Start declaring a pass; passes run in dependency order, derived from the resources they use.")
        .def("image", &FrameGraph::image, "name"_a, "Handle for a named image (\"swapchain\" is the window).")
        .def("buffer", &FrameGraph::buffer, "name"_a, "Handle for a named buffer.")
        .def("compile", &FrameGraph::compile,
             "Build the graph now. Rarely needed: run() compiles, and changes while running (new or modified "
             "passes, replaced uniform/storage buffers or sampled images) recompile automatically.")
        .def_prop_ro("needs_recompile", &FrameGraph::needsRecompile)
        .def_prop_ro("compile_count", &FrameGraph::compileCount, "How many times the graph has been compiled.")
        .def("debug_dump", &FrameGraph::debugDump);
}

void bindViewer(nb::module_ &m)
{
    nb::class_<Viewer>(m, "Viewer", "Window + Vulkan device + frame graph. Declare passes, then call run().")
        .def(
            "__init__",
            [](Viewer *self, const std::string &title, int width, int height, bool validation,
               bool raiseValidationErrors) {
                new (self) Viewer(Viewer::Config{
                    .title            = title,
                    .width            = width,
                    .height           = height,
                    .enableValidation = validation,
                });
                if (validation && raiseValidationErrors)
                {
                    self->onValidationError([self](std::string_view message) {
                        g_callbackErrors.recordValidationError(*self, message);
                    });
                }
            },
            "title"_a = "lr", "width"_a = 1600, "height"_a = 900, "validation"_a = true,
            "raise_validation_errors"_a = true,
            "With validation on (the default), a validation-layer error closes the window and run() raises "
            "VulkanValidationError; pass raise_validation_errors=False to only log them.")
        .def_prop_ro("frame_graph", &Viewer::frameGraph, nb::rv_policy::reference_internal)
        .def_prop_ro("resources", &Viewer::resources, nb::rv_policy::reference_internal)
        .def_prop_ro("swapchain_format", &Viewer::swapchainFormat)
        .def(
            "on_update",
            [](Viewer &v, nb::callable callback) {
                v.onUpdate([slot = g_callbackSlots.hold(std::move(callback))](float dt, VkExtent2D extent) {
                    if (*slot)
                    {
                        g_callbackErrors.guard([&] {
                            (*slot)(dt, toTuple(extent));
                        });
                    }
                });
            },
            "callback"_a, "callback(dt: float, extent: (width, height)), called every frame before rendering.")
        .def(
            "on_late_update",
            [](Viewer &v, nb::callable callback) {
                v.onLateUpdate([slot = g_callbackSlots.hold(std::move(callback))](float dt, VkExtent2D extent) {
                    if (*slot)
                    {
                        g_callbackErrors.guard([&] {
                            (*slot)(dt, toTuple(extent));
                        });
                    }
                });
            },
            "callback"_a, "Like on_update, but after every on_update callback has run.")
        .def(
            "run",
            [](Viewer &v) {
                // The ImGui pass composites onto the swapchain, so it must come after every user pass.
                if (!v.hasImguiPass())
                {
                    v.addImguiPass();
                }
                // Validation errors left over from a different (e.g. already destroyed) Viewer don't belong to this
                // run.
                if (g_callbackErrors.validationSource != &v)
                {
                    g_callbackErrors.clearValidationErrors();
                }
                g_callbackErrors.running = &v;
                try
                {
                    v.run();
                } catch (...)
                {
                    g_callbackErrors.running = nullptr;
                    g_callbackErrors.error.reset();
                    g_callbackErrors.clearValidationErrors();
                    g_callbackSlots.releaseAll();
                    throw;
                }
                g_callbackErrors.running = nullptr;
                // The window is closed, so no callback can fire again.
                g_callbackSlots.releaseAll();
                g_callbackErrors.rethrowPending(v);
            },
            "Compile the frame graph and run until the window closes. An exception raised in any callback "
            "closes the window and is re-raised here. A Viewer runs once: its callbacks are released on return.")
        .def("close", &Viewer::requestClose, "Ask run() to return after the current frame.");
}

#if defined(_MSC_VER) && defined(_DEBUG)
// Debug builds of this module run on the debug C runtime, separate from the interpreter's. By default
// it reports fatal errors (failed checked-iterator/STL assertions, abort(), std::terminate) in modal
// dialogs, which freeze a script until someone clicks them away, and its abort() raises SIGABRT in
// its own signal table, which the interpreter's faulthandler never sees. So: reports go to stderr, and
// fatal paths end in the *interpreter's* abort(), letting faulthandler print the Python traceback.
// Release builds share the interpreter's runtime and need none of this.

[[noreturn]] void abortThroughInterpreterRuntime() noexcept
{
    std::fflush(stderr);
    using AbortFn = void (*)();
    if (HMODULE ucrt = GetModuleHandleW(L"ucrtbase.dll"))
    {
        if (const auto hostAbort = reinterpret_cast<AbortFn>(GetProcAddress(ucrt, "abort")))
        {
            hostAbort();
        }
    }
    std::abort();
}

void onTerminate() noexcept
{
    std::fputs("lr: std::terminate called", stderr);
    if (const std::exception_ptr pending = std::current_exception())
    {
        try
        {
            std::rethrow_exception(pending);
        } catch (const std::exception &e)
        {
            std::fprintf(stderr, " after an uncaught exception: %s", e.what());
        } catch (...)
        {
            std::fputs(" after an uncaught non-std exception", stderr);
        }
    }
    std::fputs("\n", stderr);
    abortThroughInterpreterRuntime();
}

void onInvalidParameter(const wchar_t *, const wchar_t *, const wchar_t *, unsigned int, uintptr_t)
{
    std::fputs("lr: C runtime check failed (see the report above)\n", stderr);
    abortThroughInterpreterRuntime();
}

// Failed assertions and checked-iterator errors are reported here first; after the report, recent
// MSVC STLs __fastfail without consulting the invalid-parameter handler, which skips faulthandler.
int onDebugReport(int reportType, char *message, int *returnValue)
{
    if (reportType == _CRT_WARN)
    {
        return FALSE; // default handling (stderr, per the report mode below)
    }
    std::fprintf(stderr, "lr: %s", message ? message : "C runtime assertion failed\n");
    *returnValue = 0;
    abortThroughInterpreterRuntime();
}

void routeDebugRuntimeErrorsToStderr()
{
    for (const int reportType : {_CRT_WARN, _CRT_ASSERT, _CRT_ERROR})
    {
        _CrtSetReportMode(reportType, _CRTDBG_MODE_FILE);
        _CrtSetReportFile(reportType, _CRTDBG_FILE_STDERR);
    }
    _CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, onDebugReport);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _set_invalid_parameter_handler(onInvalidParameter);
    std::set_terminate(onTerminate);
}
#endif

// Failure injection for lr's own tests (tests/python/test_failures.py): paths that correct Python
// code can't reach, such as a C++ exception thrown inside the frame loop or a hard crash. Not API.
void bindTesting(nb::module_ &m)
{
    nb::module_ testing = m.def_submodule("_testing", "Failure injection for lr's own tests. Not part of the API.");
#if defined(_DEBUG)
    testing.attr("DEBUG_BUILD") = true;
#else
    testing.attr("DEBUG_BUILD") = false;
#endif

    testing.def(
        "throw_in_frame_loop",
        [](Viewer &viewer, int afterFrames) {
            // A C++ (not Python) callback, so the exception escapes Viewer::run() mid-frame — after
            // the swapchain image is acquired, with earlier frames possibly still on the GPU.
            viewer.onLateUpdate([frames = std::make_shared<int>(0), afterFrames](float, VkExtent2D) {
                if (++*frames > afterFrames)
                {
                    throw std::runtime_error("lr._testing: injected C++ failure inside the frame loop");
                }
            });
        },
        "viewer"_a, "after_frames"_a);

    testing.def(
        "crash",
        [](const std::string &kind) {
            if (kind == "access_violation")
            {
                volatile int *null = nullptr;
                *null              = 1;
            } else if (kind == "debug_assert")
            {
                // Out of range: a checked-iterator assertion in MSVC Debug builds.
                std::vector<int> empty;
                volatile size_t  index = 1;
                (void)empty[index];
            } else if (kind == "terminate")
            {
                std::terminate();
            }
            throw std::invalid_argument("crash: unknown kind '" + kind + "'");
        },
        "kind"_a);
}

} // namespace

NB_MODULE(_lr, m)
{
    m.doc() = "Python frontend for the lightweight renderer's frame graph.";

#if defined(_MSC_VER) && defined(_DEBUG)
    routeDebugRuntimeErrorsToStderr();
#endif

    nb::exception<lr::ShaderCompileError>(m, "ShaderCompileError");
    nb::exception<lr::ShaderInterfaceError>(m, "ShaderInterfaceError", PyExc_RuntimeError);
    nb::exception<VulkanValidationError>(m, "VulkanValidationError", PyExc_RuntimeError);

    bindEnums(m);
    bindValueTypes(m);
    bindResources(m);
    bindPasses(m);
    bindViewer(m);
    bindTesting(m);

    m.def(
        "compile_glsl",
        [](const fs::path &path, std::optional<ShaderStage> stage, const std::vector<fs::path> &includeDirs) {
            return spirvToBytes(lr::compileGlslFile(path, stage, includeDirs));
        },
        "path"_a, "stage"_a = nb::none(), "include_dirs"_a = std::vector<fs::path>{},
        "Compile a GLSL file to SPIR-V bytes. Stage is inferred from .vert/.frag/.comp unless given.");
    m.def(
        "compile_glsl_source",
        [](const std::string &source, ShaderStage stage, const std::string &name,
           const std::vector<fs::path> &includeDirs) {
            return spirvToBytes(lr::compileGlslSource(source, stage, name, includeDirs));
        },
        "source"_a, "stage"_a, "name"_a = "<source>", "include_dirs"_a = std::vector<fs::path>{},
        "Compile GLSL source text to SPIR-V bytes.");

    m.attr("SHADER_DIR") = lr::paths::shaderDir;
    m.attr("ASSET_DIR")  = lr::paths::assetDir;

    // Covers scripts that never reach run() (or exit mid-way): drop the callbacks while the
    // interpreter can still destroy the Viewers they keep alive.
    nb::module_::import_("atexit").attr("register")(nb::cpp_function([] {
        g_callbackSlots.releaseAll();
    }));
}
