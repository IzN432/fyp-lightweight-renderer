// Python bindings for the frame graph: Viewer, ResourceRegistry, FrameGraph, PassBuilder,
// PassContext and CommandBuffer, plus runtime GLSL compilation. Built as `lr._lr` and re-exported
// by the pure-Python `lr` package (python/lr/__init__.py).

#include "core/Paths.hpp"
#include "core/app/InputHandler.hpp"
#include "core/app/Viewer.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneAssets.hpp"
#include "core/scene/SceneSerializer.hpp"
#include "core/scene/SceneGpu.hpp"
#include "core/scene/TransformComponent.hpp"
#include "core/passes/ambientocclusion/AmbientOcclusionPass.hpp"
#include "core/passes/composite/CompositePass.hpp"
#include "core/passes/geometry/GeometryPass.hpp"
#include "core/passes/ibl/IblPass.hpp"
#include "core/passes/pbr/PbrPass.hpp"
#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/PassBuilder.hpp"
#include "core/framegraph/PassContext.hpp"
#include "core/framegraph/ResourceRegistry.hpp"
#include "core/framegraph/compiler/ShaderInterface.hpp"
#include "core/vulkan/CommandBuffer.hpp"
#include "core/vulkan/ShaderCompiler.hpp"
#include "features/animation/AnimatorComponent.hpp"

#include <imgui.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unique_ptr.h>
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
#include <cassert>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <typeindex>
#include <optional>
#include <unordered_map>

// SceneObject stores its components as unique_ptrs in an unordered_map, whose copy constructor isn't
// constrained — so std::is_copy_constructible wrongly reports it copyable. Tell nanobind it isn't.
template <> struct nanobind::detail::is_copy_constructible<lr::SceneObject> : std::false_type
{};

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
using lr::InputHandler;
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

    std::shared_ptr<nb::object> hold(nb::object object)
    {
        auto slot = std::make_shared<nb::object>(std::move(object));
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

// lr supports one Viewer at a time (ImGui's context is process-wide). Each Viewer owns a token, in a
// callback it never calls, so the count drops when the C++ Viewer is destroyed.
int g_liveViewers = 0;

struct ViewerToken
{
    ViewerToken() { ++g_liveViewers; }
    ~ViewerToken() { --g_liveViewers; }
    ViewerToken(const ViewerToken &)            = delete;
    ViewerToken &operator=(const ViewerToken &) = delete;
};

// For objects the Viewer's frame loop calls into (SceneGpu's update callbacks, engine passes' execute
// callbacks): the Viewer holds `self` the way it holds Python callbacks — released when run() returns
// or at exit — rather than through keep_alive, since these objects keep the Viewer alive in turn.
template <typename T> void holdWhileViewerRuns(Viewer &viewer, T *self)
{
    nb::object object = nb::find(self);
    if (!object.is_valid())
    {
        throw std::logic_error("lr: holdWhileViewerRuns called on an unregistered instance");
    }
    viewer.ownConnection(
        viewer.onLateUpdate([slot = g_callbackSlots.hold(std::move(object))](float, VkExtent2D) {}));
}

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

// Every format lr.Format exposes.
constexpr std::pair<const char *, VkFormat> kFormats[] = {
    {"UNDEFINED", VK_FORMAT_UNDEFINED},
    {"R8_UNORM", VK_FORMAT_R8_UNORM},
    {"R8G8B8A8_UNORM", VK_FORMAT_R8G8B8A8_UNORM},
    {"R8G8B8A8_SRGB", VK_FORMAT_R8G8B8A8_SRGB},
    {"B8G8R8A8_UNORM", VK_FORMAT_B8G8R8A8_UNORM},
    {"B8G8R8A8_SRGB", VK_FORMAT_B8G8R8A8_SRGB},
    {"R16_SFLOAT", VK_FORMAT_R16_SFLOAT},
    {"R16G16_SFLOAT", VK_FORMAT_R16G16_SFLOAT},
    {"R16G16B16A16_SFLOAT", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"R16G16B16A16_UNORM", VK_FORMAT_R16G16B16A16_UNORM},
    {"R32_SFLOAT", VK_FORMAT_R32_SFLOAT},
    {"R32G32_SFLOAT", VK_FORMAT_R32G32_SFLOAT},
    {"R32G32B32_SFLOAT", VK_FORMAT_R32G32B32_SFLOAT},
    {"R32G32B32A32_SFLOAT", VK_FORMAT_R32G32B32A32_SFLOAT},
    {"R32_UINT", VK_FORMAT_R32_UINT},
    {"D32_SFLOAT", VK_FORMAT_D32_SFLOAT},
};

void bindEnums(nb::module_ &m)
{
    auto format = nb::enum_<VkFormat>(m, "Format");
    for (const auto &[name, value] : kFormats)
    {
        format.value(name, value);
    }

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
            "upload_array_image",
            [](ResourceRegistry &r, const std::string &arrayName, uint32_t index, HostArray data, VkFormat format,
               bool generateMipmaps) {
                const auto [width, height] = imageSize(data, format, "upload_array_image");
                r.uploadArrayImage(arrayName, index, data.data(), width, height, format, generateMipmaps);
            },
            "array_name"_a, "index"_a, "data"_a, "format"_a, "generate_mipmaps"_a = false,
            "Upload one element of an image array (elements may differ in size), for sampled_image_array(): "
            "how a single pass picks a different texture per draw, as the engine's GeometryPass does.")
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
                std::vector<std::byte> contents;
                {
                    nb::gil_scoped_release release; // waits for the GPU
                    contents = r.readBuffer(name);
                }
                auto       *bytes = new std::vector<std::byte>(std::move(contents));
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
        .def(
            "samples",
            [](PassBuilder &b, uint32_t count) -> PassBuilder & {
                switch (count)
                {
                    case 1: return b.samples(VK_SAMPLE_COUNT_1_BIT);
                    case 2: return b.samples(VK_SAMPLE_COUNT_2_BIT);
                    case 4: return b.samples(VK_SAMPLE_COUNT_4_BIT);
                    case 8: return b.samples(VK_SAMPLE_COUNT_8_BIT);
                    case 16: return b.samples(VK_SAMPLE_COUNT_16_BIT);
                    case 32: return b.samples(VK_SAMPLE_COUNT_32_BIT);
                    case 64: return b.samples(VK_SAMPLE_COUNT_64_BIT);
                    default: throw std::invalid_argument("samples: count must be 1, 2, 4, 8, 16, 32, or 64");
                }
            },
            "count"_a, ref, "Enable MSAA for this pass; attachments are resolved automatically.")
        .def("polygon_mode", &PassBuilder::polygonMode, "mode"_a, ref, "FILL (default), LINE (wireframe) or POINT.")
        .def("cull", &PassBuilder::cull, "mode"_a, "front_face"_a = VK_FRONT_FACE_COUNTER_CLOCKWISE, ref,
             "Default: BACK for geometry passes, NONE for fullscreen ones.")
        .def("depth", &PassBuilder::depth, "test"_a, "write"_a, "compare"_a = VK_COMPARE_OP_LESS, ref,
             "Default: test and write on (compare LESS) exactly when the pass has a depth attachment.")
        .def("depth_bias", &PassBuilder::depthBias, "constant"_a, "slope"_a = 0.0f, ref,
             "Offset rasterized depth, e.g. negative values to draw a wireframe over its own solid surface.")
        .def("sampled_image", &PassBuilder::sampledImage, "binding"_a, "image"_a, "stages"_a, ref)
        .def("sampled_depth", &PassBuilder::sampledDepth, "binding"_a, "image"_a, "stages"_a, ref)
        .def("sampled_image_array", &PassBuilder::sampledImageArray, "binding"_a, "images"_a, "count"_a, "stages"_a,
             ref,
             "Bind the first `count` elements of an image array (see ResourceRegistry.upload_array_image) as "
             "`uniform sampler2D name[count]`.")
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
        .def("image", &FrameGraph::image, "name"_a, "Handle for a named image (lr.SWAPCHAIN is the window).")
        .def("buffer", &FrameGraph::buffer, "name"_a, "Handle for a named buffer.")
        .def("compile", &FrameGraph::compile,
             "Build the graph now. Rarely needed: run() compiles, and changes while running (new or modified "
             "passes, replaced uniform/storage buffers or sampled images) recompile automatically.")
        .def_prop_ro("needs_recompile", &FrameGraph::needsRecompile)
        .def_prop_ro("compile_count", &FrameGraph::compileCount, "How many times the graph has been compiled.")
        .def("debug_dump", &FrameGraph::debugDump);
}

// GLFW key/button codes, as Python enums.
enum class Key : int
{
};
enum class MouseButton : int
{
    Left   = GLFW_MOUSE_BUTTON_LEFT,
    Right  = GLFW_MOUSE_BUTTON_RIGHT,
    Middle = GLFW_MOUSE_BUTTON_MIDDLE,
};

void bindInput(nb::module_ &m)
{
    auto key = nb::enum_<Key>(m, "Key", "Keyboard keys (GLFW key codes).");
    // nanobind keeps the name pointers, so they live in static storage.
    static const std::array<std::string, 26> letters = [] {
        std::array<std::string, 26> names;
        for (int i = 0; i < 26; ++i)
        {
            names[i] = std::string(1, static_cast<char>('A' + i));
        }
        return names;
    }();
    static const std::array<std::string, 10> digits = [] {
        std::array<std::string, 10> names;
        for (int i = 0; i < 10; ++i)
        {
            names[i] = "DIGIT_" + std::to_string(i);
        }
        return names;
    }();
    static const std::array<std::string, 12> functionKeys = [] {
        std::array<std::string, 12> names;
        for (int i = 0; i < 12; ++i)
        {
            names[i] = "F" + std::to_string(i + 1);
        }
        return names;
    }();
    for (int i = 0; i < 26; ++i)
    {
        key.value(letters[i].c_str(), static_cast<Key>(GLFW_KEY_A + i));
    }
    for (int i = 0; i < 10; ++i)
    {
        key.value(digits[i].c_str(), static_cast<Key>(GLFW_KEY_0 + i));
    }
    for (int i = 0; i < 12; ++i)
    {
        key.value(functionKeys[i].c_str(), static_cast<Key>(GLFW_KEY_F1 + i));
    }
    key.value("SPACE", static_cast<Key>(GLFW_KEY_SPACE))
        .value("ESCAPE", static_cast<Key>(GLFW_KEY_ESCAPE))
        .value("ENTER", static_cast<Key>(GLFW_KEY_ENTER))
        .value("TAB", static_cast<Key>(GLFW_KEY_TAB))
        .value("BACKSPACE", static_cast<Key>(GLFW_KEY_BACKSPACE))
        .value("DELETE", static_cast<Key>(GLFW_KEY_DELETE))
        .value("LEFT", static_cast<Key>(GLFW_KEY_LEFT))
        .value("RIGHT", static_cast<Key>(GLFW_KEY_RIGHT))
        .value("UP", static_cast<Key>(GLFW_KEY_UP))
        .value("DOWN", static_cast<Key>(GLFW_KEY_DOWN))
        .value("LEFT_SHIFT", static_cast<Key>(GLFW_KEY_LEFT_SHIFT))
        .value("RIGHT_SHIFT", static_cast<Key>(GLFW_KEY_RIGHT_SHIFT))
        .value("LEFT_CONTROL", static_cast<Key>(GLFW_KEY_LEFT_CONTROL))
        .value("RIGHT_CONTROL", static_cast<Key>(GLFW_KEY_RIGHT_CONTROL))
        .value("LEFT_ALT", static_cast<Key>(GLFW_KEY_LEFT_ALT))
        .value("RIGHT_ALT", static_cast<Key>(GLFW_KEY_RIGHT_ALT));

    nb::enum_<MouseButton>(m, "MouseButton")
        .value("LEFT", MouseButton::Left)
        .value("RIGHT", MouseButton::Right)
        .value("MIDDLE", MouseButton::Middle);

    nb::class_<InputHandler>(m, "Input",
                             "Keyboard and mouse state, updated once per frame before on_update. It reports the "
                             "raw state even over UI; check lr.gui.want_capture_mouse() to leave the mouse to "
                             "ImGui.")
        .def(
            "is_key_down",
            [](const InputHandler &input, Key key) {
                return input.isKeyPressed(static_cast<int>(key));
            },
            "key"_a)
        .def(
            "is_mouse_down",
            [](const InputHandler &input, MouseButton button) {
                return input.isMouseButtonPressed(static_cast<int>(button));
            },
            "button"_a)
        .def_prop_ro(
            "mouse_position",
            [](const InputHandler &input) {
                double x = 0.0, y = 0.0;
                input.getMousePos(x, y);
                return nb::make_tuple(x, y);
            },
            "Cursor position in window pixels, (x, y).")
        .def_prop_ro(
            "mouse_delta",
            [](const InputHandler &input) {
                double dx = 0.0, dy = 0.0;
                input.getMouseDelta(dx, dy);
                return nb::make_tuple(dx, dy);
            },
            "Cursor movement since the previous frame, (dx, dy) in pixels.")
        .def_prop_ro("scroll_delta", &InputHandler::getScrollDelta, "Scroll-wheel movement since the previous frame.")
        .def_prop_ro("shift", &InputHandler::isShiftPressed)
        .def_prop_ro("ctrl", &InputHandler::isCtrlPressed)
        .def_prop_ro("alt", &InputHandler::isAltPressed);
}

// ImGui from Python. Widgets only work between the frame's ImGui NewFrame and Render, i.e. inside an
// on_gui callback; outside it ImGui would assert (aborting the interpreter), so they raise instead.
// Windows left open (an exception inside `with gui.window(...)`, or a missing end()) are closed when
// the callback returns, for the same reason.
struct GuiScope
{
    bool active      = false;
    int  openWindows = 0;
};
GuiScope g_gui;

void requireGui(const char *function)
{
    if (!g_gui.active)
    {
        throw std::logic_error(std::string("lr.gui.") + function +
                               "() can only be called from a Viewer.on_gui callback");
    }
}

void bindGui(nb::module_ &m)
{
    nb::module_ gui = m.def_submodule("_gui", "ImGui subset; use it through lr.gui.");

    gui.def(
        "begin",
        [](const std::string &title, bool closable, std::optional<std::array<float, 2>> size,
           std::optional<std::array<float, 2>> position) {
            requireGui("begin");
            if (title.empty())
            {
                throw std::invalid_argument("lr.gui.window(): title must not be empty");
            }
            // Only the first time the window appears; afterwards the user's resizing/moving wins.
            if (size)
            {
                ImGui::SetNextWindowSize(ImVec2((*size)[0], (*size)[1]), ImGuiCond_FirstUseEver);
            }
            if (position)
            {
                ImGui::SetNextWindowPos(ImVec2((*position)[0], (*position)[1]), ImGuiCond_FirstUseEver);
            }
            bool open    = true;
            bool visible = ImGui::Begin(title.c_str(), closable ? &open : nullptr);
            ++g_gui.openWindows;
            return nb::make_tuple(visible, open);
        },
        "title"_a, "closable"_a = false, "size"_a = nb::none(), "position"_a = nb::none());
    gui.def("end", [] {
        requireGui("end");
        if (g_gui.openWindows == 0)
        {
            throw std::logic_error("lr.gui.end() without a matching begin()");
        }
        ImGui::End();
        --g_gui.openWindows;
    });
    gui.def(
        "text",
        [](const std::string &text) {
            requireGui("text");
            ImGui::TextUnformatted(text.c_str());
        },
        "text"_a);
    gui.def(
        "button",
        [](const std::string &label) {
            requireGui("button");
            return ImGui::Button(label.c_str());
        },
        "label"_a);
    gui.def(
        "checkbox",
        [](const std::string &label, bool value) {
            requireGui("checkbox");
            const bool changed = ImGui::Checkbox(label.c_str(), &value);
            return nb::make_tuple(changed, value);
        },
        "label"_a, "value"_a);
    gui.def(
        "slider_float",
        [](const std::string &label, float value, float min, float max, const std::string &format, bool logarithmic) {
            requireGui("slider_float");
            const bool changed = ImGui::SliderFloat(label.c_str(), &value, min, max, format.c_str(),
                                                    logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
            return nb::make_tuple(changed, value);
        },
        "label"_a, "value"_a, "min"_a, "max"_a, "format"_a = "%.3f", "logarithmic"_a = false);
    gui.def(
        "slider_int",
        [](const std::string &label, int value, int min, int max) {
            requireGui("slider_int");
            const bool changed = ImGui::SliderInt(label.c_str(), &value, min, max);
            return nb::make_tuple(changed, value);
        },
        "label"_a, "value"_a, "min"_a, "max"_a);
    gui.def(
        "drag_float",
        [](const std::string &label, float value, float speed, float min, float max) {
            requireGui("drag_float");
            const bool changed = ImGui::DragFloat(label.c_str(), &value, speed, min, max);
            return nb::make_tuple(changed, value);
        },
        "label"_a, "value"_a, "speed"_a = 0.01f, "min"_a = 0.0f, "max"_a = 0.0f);
    gui.def(
        "color_edit3",
        [](const std::string &label, std::array<float, 3> color) {
            requireGui("color_edit3");
            const bool changed = ImGui::ColorEdit3(label.c_str(), color.data());
            return nb::make_tuple(changed, nb::make_tuple(color[0], color[1], color[2]));
        },
        "label"_a, "color"_a);
    gui.def(
        "color_edit4",
        [](const std::string &label, std::array<float, 4> color) {
            requireGui("color_edit4");
            const bool changed = ImGui::ColorEdit4(label.c_str(), color.data());
            return nb::make_tuple(changed, nb::make_tuple(color[0], color[1], color[2], color[3]));
        },
        "label"_a, "color"_a);
    gui.def(
        "combo",
        [](const std::string &label, int current, const std::vector<std::string> &items) {
            requireGui("combo");
            if (items.empty())
            {
                throw std::invalid_argument("lr.gui.combo(): items must not be empty");
            }
            std::vector<const char *> names;
            names.reserve(items.size());
            for (const std::string &item : items)
            {
                names.push_back(item.c_str());
            }
            const bool changed = ImGui::Combo(label.c_str(), &current, names.data(), static_cast<int>(names.size()));
            return nb::make_tuple(changed, current);
        },
        "label"_a, "current"_a, "items"_a);
    gui.def(
        "collapsing_header",
        [](const std::string &label, bool defaultOpen) {
            requireGui("collapsing_header");
            return ImGui::CollapsingHeader(label.c_str(), defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen : 0);
        },
        "label"_a, "default_open"_a = true);
    gui.def("separator", [] {
        requireGui("separator");
        ImGui::Separator();
    });
    gui.def("same_line", [] {
        requireGui("same_line");
        ImGui::SameLine();
    });
    gui.def("spacing", [] {
        requireGui("spacing");
        ImGui::Spacing();
    });
    // Safe at any time once a Viewer exists (its ImGui context is created with it).
    gui.def("want_capture_mouse", [] {
        return ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse;
    });
    gui.def("want_capture_keyboard", [] {
        return ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard;
    });
    gui.def("framerate", [] {
        return ImGui::GetCurrentContext() ? ImGui::GetIO().Framerate : 0.0f;
    });
}

// The engine's own orbit camera, as the C++ renderer uses it: a Camera + TransformComponent on a scene
// object, driven by SphericalCameraController. Same controls, same maths, same projection.
class OrbitCamera
{
public:
    explicit OrbitCamera(Viewer &viewer)
        : m_object(&m_scene.createSceneObject()), m_camera(&m_object->addComponent<lr::Camera>()),
          m_input(&viewer.input())
    {
        m_object->addComponent<lr::TransformComponent>();
        m_controller = &m_object->addComponent<lr::SphericalCameraController>();
        m_controller->setOrbitState(m_controller->orbitState()); // place the camera before the first update
    }

    OrbitCamera(const OrbitCamera &) = delete;
    OrbitCamera &operator=(const OrbitCamera &) = delete;

    lr::SphericalCameraController &controller() { return *m_controller; }
    lr::Camera                    &camera() { return *m_camera; }
    lr::SceneObject               &object() { return *m_object; }
    glm::vec3 position() const { return m_object->getComponent<lr::TransformComponent>().transform().position(); }
    void update(float dt) { m_controller->update(*m_input, dt); }

private:
    lr::Scene                                      m_scene; // owns the camera object; declared first
    lr::SceneObject                               *m_object;
    lr::Camera                                    *m_camera;
    lr::InputHandler                              *m_input;
    lr::SphericalCameraController                 *m_controller;
};

// glm is column-major; numpy matrices here follow lr.transforms: row-major, acting on column vectors.
nb::object toNumpy(const glm::mat4 &matrix)
{
    auto *rows = new float[16];
    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 4; ++column)
        {
            rows[row * 4 + column] = matrix[column][row];
        }
    }
    nb::capsule owner(rows, [](void *p) noexcept {
        delete[] static_cast<float *>(p);
    });
    return nb::cast(nb::ndarray<nb::numpy, float, nb::shape<4, 4>>(rows, {4, 4}, owner));
}

void bindCamera(nb::module_ &m)
{
    using State              = lr::SphericalCameraController::OrbitState;
    const auto orbitProperty = [](auto field) {
        return [field](OrbitCamera &camera, float value) {
            State state  = camera.controller().orbitState();
            state.*field = value;
            camera.controller().setOrbitState(state);
        };
    };

    nb::class_<OrbitCamera>(m, "OrbitCamera",
                            "The engine's orbit camera (SphericalCameraController + Camera), with the C++ "
                            "renderer's controls: middle-drag orbits, Shift + middle-drag pans, the scroll wheel "
                            "zooms and R resets. Ignores the mouse while it is over ImGui UI. Call update() once "
                            "per frame, then use matrices(extent) for a `mat4 view; mat4 proj;` uniform block.")
        .def(nb::init<Viewer &>(), "viewer"_a, nb::keep_alive<1, 2>())
        .def(
            "update",
            [](OrbitCamera &camera, float dt) {
                camera.update(dt);
            },
            "dt"_a, "Apply this frame's input (from on_update).")
        .def(
            "view_matrix",
            [](OrbitCamera &camera) {
                return toNumpy(camera.camera().viewMatrix());
            },
            "4x4 float32 view matrix (row-major numpy, acting on column vectors).")
        .def(
            "projection_matrix",
            [](OrbitCamera &camera, float aspect) {
                return toNumpy(camera.camera().projectionMatrix(aspect));
            },
            "aspect"_a, "4x4 float32 projection: Vulkan clip space, depth in [0, 1], Y flipped — as the engine uses.")
        .def(
            "matrices",
            [](OrbitCamera &camera, std::pair<uint32_t, uint32_t> extent) {
                const float aspect = static_cast<float>(extent.first) / static_cast<float>(std::max(extent.second, 1u));
                auto *data = new glm::mat4[2]{camera.camera().viewMatrix(), camera.camera().projectionMatrix(aspect)};
                nb::capsule owner(data, [](void *p) noexcept {
                    delete[] static_cast<glm::mat4 *>(p);
                });
                return nb::ndarray<nb::numpy, float, nb::shape<32>>(data, {32}, owner);
            },
            "extent"_a, "view and projection for a (width, height) extent, packed for a `mat4 view; mat4 proj;` block.")
        .def_prop_ro(
            "position",
            [](const OrbitCamera &camera) {
                const glm::vec3 p = camera.position();
                return nb::make_tuple(p.x, p.y, p.z);
            },
            "Camera position in world space.")
        .def_prop_rw(
            "target",
            [](OrbitCamera &camera) {
                const glm::vec3 t = camera.controller().orbitState().target;
                return nb::make_tuple(t.x, t.y, t.z);
            },
            [](OrbitCamera &camera, std::array<float, 3> target) {
                State state  = camera.controller().orbitState();
                state.target = glm::vec3(target[0], target[1], target[2]);
                camera.controller().setOrbitState(state);
            },
            "The point orbited around.")
        .def_prop_rw(
            "radius",
            [](OrbitCamera &camera) {
                return camera.controller().orbitState().radius;
            },
            orbitProperty(&State::radius), "Distance from the target (clamped to [0.01, 1000]).")
        .def_prop_rw(
            "azimuth",
            [](OrbitCamera &camera) {
                return camera.controller().orbitState().azimuth;
            },
            orbitProperty(&State::azimuth), "Radians around +Y; 0 places the camera on the target's +Z side.")
        .def_prop_rw(
            "elevation",
            [](OrbitCamera &camera) {
                return camera.controller().orbitState().elevation;
            },
            orbitProperty(&State::elevation), "Radians above the horizontal (clamped to ±89°).")
        .def_prop_rw(
            "fov_y_degrees",
            [](OrbitCamera &camera) {
                return camera.camera().fovYDegrees;
            },
            [](OrbitCamera &camera, float value) {
                camera.camera().fovYDegrees = value;
            })
        .def_prop_rw(
            "near_plane",
            [](OrbitCamera &camera) {
                return camera.camera().nearPlane;
            },
            [](OrbitCamera &camera, float value) {
                camera.camera().nearPlane = value;
            })
        .def_prop_rw(
            "far_plane",
            [](OrbitCamera &camera) {
                return camera.camera().farPlane;
            },
            [](OrbitCamera &camera, float value) {
                camera.camera().farPlane = value;
            })
        .def_prop_rw(
            "orthographic",
            [](OrbitCamera &camera) {
                return camera.camera().projectionType == lr::ProjectionType::Orthographic;
            },
            [](OrbitCamera &camera, bool value) {
                camera.camera().projectionType =
                    value ? lr::ProjectionType::Orthographic : lr::ProjectionType::Perspective;
            },
            "Orthographic instead of perspective projection (height set by ortho_height).")
        .def_prop_rw(
            "ortho_height",
            [](OrbitCamera &camera) {
                return camera.camera().orthoHeight;
            },
            [](OrbitCamera &camera, float value) {
                camera.camera().orthoHeight = value;
            });
}

// ---------------------------------------------------------------------------------------------
// Scenes (CPU side): what the engine's loaders produce, as numpy
// ---------------------------------------------------------------------------------------------

// A numpy array that owns a copy of `data`.
// Returned as a plain object: property getters default to reference_internal, which doesn't apply to an
// array that owns its data.
template <typename T> nb::object ownedArray(std::vector<T> data, std::vector<size_t> shape)
{
    auto       *storage = new std::vector<T>(std::move(data));
    nb::capsule owner(storage, [](void *p) noexcept {
        delete static_cast<std::vector<T> *>(p);
    });
    return nb::cast(nb::ndarray<nb::numpy, T>(storage->data(), shape.size(), shape.data(), owner));
}

template <typename T> nb::object rawToNumpy(std::span<const std::byte> raw, size_t count, size_t components)
{
    std::vector<T> values(count * components);
    std::memcpy(values.data(), raw.data(), std::min(raw.size(), values.size() * sizeof(T)));
    return components == 1 ? ownedArray(std::move(values), {count})
                           : ownedArray(std::move(values), {count, components});
}

// A mesh attribute as numpy: known element types become (count, components) arrays of their scalar
// type; anything else comes back as raw bytes, (count, stride) uint8.
nb::object attributeToNumpy(std::span<const std::byte> raw, const lr::MeshLayout::AttributeDesc &desc, size_t count)
{
    const std::type_index type = desc.type;
    if (type == typeid(float))
    {
        return rawToNumpy<float>(raw, count, 1);
    }
    if (type == typeid(glm::vec2))
    {
        return rawToNumpy<float>(raw, count, 2);
    }
    if (type == typeid(glm::vec3))
    {
        return rawToNumpy<float>(raw, count, 3);
    }
    if (type == typeid(glm::vec4))
    {
        return rawToNumpy<float>(raw, count, 4);
    }
    if (type == typeid(uint32_t))
    {
        return rawToNumpy<uint32_t>(raw, count, 1);
    }
    if (type == typeid(glm::uvec4))
    {
        return rawToNumpy<uint32_t>(raw, count, 4);
    }
    if (type == typeid(int32_t))
    {
        return rawToNumpy<int32_t>(raw, count, 1);
    }
    if (type == typeid(glm::ivec4))
    {
        return rawToNumpy<int32_t>(raw, count, 4);
    }
    return rawToNumpy<uint8_t>(raw, count, desc.stride);
}

const lr::MeshLayout::AttributeDesc &requirePerVertexAttr(const lr::Mesh &mesh, const std::string &name)
{
    const lr::MeshLayout::AttributeDesc *desc = mesh.layout().findPerVertexAttr(name);
    if (!desc)
    {
        std::string known;
        for (const auto &attr : mesh.layout().perVertexAttrs())
        {
            known += (known.empty() ? "" : ", ") + attr.name;
        }
        throw nb::key_error(("mesh has no per-vertex attribute '" + name + "' (has: " + known + ")").c_str());
    }
    return *desc;
}

lr::TransformComponent &requireTransform(lr::SceneObject &object)
{
    if (!object.hasComponent<lr::TransformComponent>())
    {
        throw std::logic_error("scene object '" + object.name + "' has no transform");
    }
    return object.getComponent<lr::TransformComponent>();
}

// A light's own parameters; where it is and which way it points come from its object's transform.
struct LightInfo
{
    std::string                         type;
    std::array<float, 3>                color{};
    float                               intensity = 0.0f;
    std::optional<float>                innerConeDegrees;
    std::optional<float>                outerConeDegrees;
    std::optional<std::array<float, 2>> areaSize;
    std::optional<bool>                 twoSided;
};

LightInfo describeLight(const lr::Light &light)
{
    LightInfo info;
    std::visit(
        [&](const auto &l) {
            using T        = std::decay_t<decltype(l)>;
            info.color     = {l.color.x, l.color.y, l.color.z};
            info.intensity = l.intensity;
            if constexpr (std::is_same_v<T, lr::PointLight>)
            {
                info.type = "point";
            } else if constexpr (std::is_same_v<T, lr::SpotLight>)
            {
                info.type             = "spot";
                info.innerConeDegrees = l.innerConeAngleDegrees;
                info.outerConeDegrees = l.outerConeAngleDegrees;
            } else if constexpr (std::is_same_v<T, lr::AreaLight>)
            {
                info.type     = "area";
                info.areaSize = std::array<float, 2>{l.size.x, l.size.y};
                info.twoSided = l.twoSided;
            } else if constexpr (std::is_same_v<T, lr::DirectionalLight>)
            {
                info.type = "directional";
            } else
            {
                info.type = "image";
            }
        },
        light.light);
    return info;
}

// Everything that defines a light apart from its transform, as Scene.add_light/SceneObject.set_light take it.
struct LightSpec
{
    std::string          type;
    std::array<float, 3> color{};
    float                intensity        = 1.0f;
    float                innerConeDegrees = 15.0f;
    float                outerConeDegrees = 30.0f;
    std::array<float, 2> size{1.0f, 1.0f};
    bool                 twoSided = true;
};

lr::LightVariant makeLight(const char *function, const LightSpec &spec)
{
    const lr::BaseLight base{glm::vec3(spec.color[0], spec.color[1], spec.color[2]), spec.intensity};
    if (spec.type == "point")
    {
        return lr::PointLight{base};
    }
    if (spec.type == "spot")
    {
        return lr::SpotLight{base, spec.innerConeDegrees, spec.outerConeDegrees};
    }
    if (spec.type == "area")
    {
        return lr::AreaLight{base, glm::vec2(spec.size[0], spec.size[1]), spec.twoSided};
    }
    if (spec.type == "directional")
    {
        return lr::DirectionalLight{base};
    }
    if (spec.type == "image")
    {
        return lr::ImageLight{base};
    }
    throw std::invalid_argument(std::string(function) +
                                ": type must be 'point', 'spot', 'area', 'directional' or 'image', not '" + spec.type +
                                "'");
}

nb::object materialValue(const lr::MaterialValue &value)
{
    return std::visit(
        [](const auto &v) -> nb::object {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, lr::MaterialParam::ColorRGBA>)
            {
                return nb::make_tuple(v.value.x, v.value.y, v.value.z, v.value.w);
            } else if constexpr (std::is_same_v<T, lr::MaterialParam::ColorRGB>)
            {
                return nb::make_tuple(v.value.x, v.value.y, v.value.z);
            } else
            {
                return nb::cast(v.value);
            }
        },
        value);
}

std::vector<lr::SceneObject *> liveObjects(lr::Scene &scene)
{
    std::vector<lr::SceneObject *> objects;
    for (const auto &object : scene.sceneObjects())
    {
        if (object && scene.contains(object->id()))
        {
            objects.push_back(object.get());
        }
    }
    return objects;
}

void bindScene(nb::module_ &m)
{
    const auto ref = nb::rv_policy::reference_internal;

    nb::class_<LightInfo>(m, "Light",
                          "A light's own parameters. Its position and orientation are its scene "
                          "object's transform (the engine's light buffer uses that object's local "
                          "position/rotation).")
        .def_ro("type", &LightInfo::type, "'point', 'spot', 'area', 'directional' or 'image'.")
        .def_prop_ro("color",
                     [](const LightInfo &light) {
                         return nb::make_tuple(light.color[0], light.color[1], light.color[2]);
                     })
        .def_ro("intensity", &LightInfo::intensity)
        .def_ro("inner_cone_degrees", &LightInfo::innerConeDegrees, "Spot lights only.")
        .def_ro("outer_cone_degrees", &LightInfo::outerConeDegrees, "Spot lights only.")
        .def_prop_ro(
            "area_size",
            [](const LightInfo &light) -> nb::object {
                if (!light.areaSize)
                {
                    return nb::none();
                }
                return nb::make_tuple((*light.areaSize)[0], (*light.areaSize)[1]);
            },
            "Area lights only: (width, height) in world units.")
        .def_ro("two_sided", &LightInfo::twoSided,
                "Area lights only: emits from both faces (True) or only along its forward axis.");

    nb::class_<lr::Material>(m, "Material",
                             "A material from the scene's material store (see EngineConventions.hpp "
                             "for the parameter and texture names the loaders use).")
        .def_ro("name", &lr::Material::name)
        .def_prop_ro(
            "parameters",
            [](const lr::Material &material) {
                nb::dict parameters;
                for (const auto &[name, value] : material.parameters)
                {
                    parameters[name.c_str()] = materialValue(value);
                }
                return parameters;
            },
            "{name: float or colour tuple}, e.g. {'baseDiffuse': (r, g, b, a), 'baseRoughness': 0.5}.")
        .def_prop_ro(
            "texture_names",
            [](const lr::Material &material) {
                std::vector<std::string> names;
                for (const auto &[name, image] : material.textures)
                {
                    if (!image.empty())
                    {
                        names.push_back(name);
                    }
                }
                std::sort(names.begin(), names.end());
                return names;
            },
            "Names of the textures this material has, e.g. ['baseColorTexture', 'normalTexture'].")
        .def(
            "texture",
            [](const lr::Material &material, const std::string &name) {
                const auto it = material.textures.find(name);
                if (it == material.textures.end() || it->second.empty())
                {
                    throw nb::key_error(("material '" + material.name + "' has no texture '" + name + "'").c_str());
                }
                const lr::MaterialImage &image    = it->second;
                const size_t             texels   = static_cast<size_t>(image.width) * image.height;
                const size_t             channels = texels ? image.pixels.size() / texels : 0;
                return ownedArray(image.pixels, {image.height, image.width, channels});
            },
            "name"_a, "The texture's pixels as a (height, width, channels) uint8 array, ready for upload_image().");

    nb::class_<lr::Mesh>(m, "Mesh",
                         "Mesh data in the engine's layout. Render vertices are corners: each has a position "
                         "(shared through position_indices) and its own attributes (normal, uv, …), so "
                         "positions/attribute()/indices are directly usable as vertex and index buffers.")
        .def_prop_ro("vertex_count", &lr::Mesh::vertexCount)
        .def_prop_ro("face_count", &lr::Mesh::faceCount)
        .def_prop_ro(
            "positions",
            [](const lr::Mesh &mesh) {
                std::vector<float> data;
                data.reserve(mesh.vertexCount() * 3);
                for (uint32_t index : mesh.positionIndices())
                {
                    const glm::vec3 &p = mesh.positions()[index];
                    data.insert(data.end(), {p.x, p.y, p.z});
                }
                return ownedArray(std::move(data), {mesh.vertexCount(), 3});
            },
            "(vertex_count, 3) float32: each render vertex's position.")
        .def_prop_ro(
            "indices",
            [](const lr::Mesh &mesh) {
                std::vector<uint32_t> data;
                data.reserve(mesh.faceCount() * 3);
                for (const glm::uvec3 &face : mesh.faces())
                {
                    data.insert(data.end(), {face.x, face.y, face.z});
                }
                return ownedArray(std::move(data), {mesh.faceCount(), 3});
            },
            "(face_count, 3) uint32 triangle vertex indices.")
        .def_prop_ro(
            "face_materials",
            [](const lr::Mesh &mesh) {
                return ownedArray(mesh.faceGroups(), {mesh.faceGroups().size()});
            },
            "(face_count,) uint32: each face's material handle (see Scene.material()); empty if the mesh has none.")
        .def_prop_ro(
            "attribute_names",
            [](const lr::Mesh &mesh) {
                std::vector<std::string> names;
                for (const auto &attr : mesh.layout().perVertexAttrs())
                {
                    names.push_back(attr.name);
                }
                return names;
            },
            "Per-vertex attributes this mesh has, e.g. ['normal', 'tangent', 'uv'].")
        .def(
            "attribute",
            [](const lr::Mesh &mesh, const std::string &name) {
                const auto &desc = requirePerVertexAttr(mesh, name);
                return attributeToNumpy(mesh.rawPerVertexData(name), desc, mesh.vertexCount());
            },
            "name"_a, "A per-vertex attribute: (vertex_count, components), e.g. 'normal' -> (V, 3) float32.")
        .def_prop_ro(
            "unique_positions",
            [](const lr::Mesh &mesh) {
                std::vector<float> data;
                data.reserve(mesh.uniquePositionCount() * 3);
                for (const glm::vec3 &p : mesh.positions())
                {
                    data.insert(data.end(), {p.x, p.y, p.z});
                }
                return ownedArray(std::move(data), {mesh.uniquePositionCount(), 3});
            },
            "(unique_position_count, 3) float32: positions shared between corners (for topology/geometry processing).")
        .def_prop_ro(
            "position_indices",
            [](const lr::Mesh &mesh) {
                return ownedArray(mesh.positionIndices(), {mesh.positionIndices().size()});
            },
            "(vertex_count,) uint32: each render vertex's index into unique_positions.");

    nb::class_<lr::AnimatorComponent>(m, "Animator", "Animation playback for an imported animated object.")
        .def_prop_ro("clip_names",
                     [](const lr::AnimatorComponent &animator) {
                         std::vector<std::string> names;
                         for (const auto &clip : animator.clips())
                         {
                             names.push_back(clip.name());
                         }
                         return names;
                     })
        .def_prop_ro("playing", &lr::AnimatorComponent::isPlaying)
        .def("play", &lr::AnimatorComponent::play, "clip_index"_a = 0)
        .def("pause", &lr::AnimatorComponent::pause)
        .def("stop", &lr::AnimatorComponent::stop)
        .def("seek", &lr::AnimatorComponent::seek, "seconds"_a);

    nb::class_<lr::SceneObject>(m, "SceneObject",
                                "An object in a Scene: a name, a place in the hierarchy, a "
                                "transform, and optionally a mesh, a light or an animator.")
        .def_prop_ro("id", &lr::SceneObject::id)
        .def_rw("name", &lr::SceneObject::name)
        .def_prop_ro(
            "parent",
            [](lr::SceneObject &object) -> lr::SceneObject * {
                return object.parent() ? &object.scene().getSceneObject(*object.parent()) : nullptr;
            },
            ref)
        .def_prop_ro(
            "children",
            [](lr::SceneObject &object) {
                std::vector<lr::SceneObject *> children;
                for (lr::SceneObjectId id : object.children())
                {
                    children.push_back(&object.scene().getSceneObject(id));
                }
                return children;
            },
            ref)
        .def_prop_ro(
            "world_matrix",
            [](const lr::SceneObject &object) {
                return toNumpy(object.worldMatrix());
            },
            "4x4 float32 object-to-world matrix (row-major numpy, acting on column vectors).")
        .def_prop_rw(
            "position",
            [](lr::SceneObject &object) {
                const glm::vec3 p = requireTransform(object).transform().position();
                return nb::make_tuple(p.x, p.y, p.z);
            },
            [](lr::SceneObject &object, std::array<float, 3> p) {
                requireTransform(object).setPosition(glm::vec3(p[0], p[1], p[2]));
            },
            "Local position (relative to the parent).")
        .def_prop_rw(
            "rotation",
            [](lr::SceneObject &object) {
                const glm::quat q = requireTransform(object).transform().rotation();
                return nb::make_tuple(q.x, q.y, q.z, q.w);
            },
            [](lr::SceneObject &object, std::array<float, 4> q) {
                requireTransform(object).setRotation(glm::quat(q[3], q[0], q[1], q[2]));
            },
            "Local rotation as a unit quaternion (x, y, z, w).")
        .def_prop_rw(
            "scale",
            [](lr::SceneObject &object) {
                const glm::vec3 s = requireTransform(object).transform().scale();
                return nb::make_tuple(s.x, s.y, s.z);
            },
            [](lr::SceneObject &object, std::array<float, 3> s) {
                requireTransform(object).setScale(glm::vec3(s[0], s[1], s[2]));
            },
            "Local scale.")
        .def_prop_ro(
            "mesh",
            [](lr::SceneObject &object) -> lr::Mesh * {
                return object.hasComponent<lr::MeshComponent>() ? &object.getComponent<lr::MeshComponent>().mesh()
                                                                : nullptr;
            },
            ref, "The object's mesh, or None.")
        .def_prop_ro(
            "light",
            [](lr::SceneObject &object) -> std::optional<LightInfo> {
                if (!object.hasComponent<lr::Light>())
                {
                    return std::nullopt;
                }
                return describeLight(object.getComponent<lr::Light>());
            },
            "The object's light, or None.")
        .def(
            "set_light",
            [](lr::SceneObject &object, std::optional<std::string> type, std::optional<std::array<float, 3>> color,
               std::optional<float> intensity, std::optional<std::array<float, 2>> size,
               std::optional<float> innerConeDegrees, std::optional<float> outerConeDegrees,
               std::optional<bool> twoSided) {
                if (!object.hasComponent<lr::Light>())
                {
                    throw std::logic_error("set_light: scene object '" + object.name + "' has no light");
                }
                lr::Light      &light   = object.getComponent<lr::Light>();
                const LightInfo current = describeLight(light);
                // Unspecified values keep the light's current ones (or the add_light defaults, for
                // parameters its current type doesn't have).
                LightSpec spec{.type      = type.value_or(current.type),
                               .color     = color.value_or(current.color),
                               .intensity = intensity.value_or(current.intensity)};
                spec.innerConeDegrees = innerConeDegrees.value_or(current.innerConeDegrees.value_or(15.0f));
                spec.outerConeDegrees = outerConeDegrees.value_or(current.outerConeDegrees.value_or(30.0f));
                spec.size             = size.value_or(current.areaSize.value_or(std::array<float, 2>{1.0f, 1.0f}));
                spec.twoSided         = twoSided.value_or(current.twoSided.value_or(true));
                light.set(makeLight("set_light", spec));
            },
            "type"_a = nb::none(), "color"_a = nb::none(), "intensity"_a = nb::none(), "size"_a = nb::none(),
            "inner_cone_degrees"_a = nb::none(), "outer_cone_degrees"_a = nb::none(), "two_sided"_a = nb::none(),
            "Change this object's light; parameters left as None keep their current values. Move or turn it "
            "with position/rotation. A SceneGpu showing the scene picks the change up on the next frame.")
        .def_prop_ro(
            "animator",
            [](lr::SceneObject &object) -> lr::AnimatorComponent * {
                return object.hasComponent<lr::AnimatorComponent>() ? &object.getComponent<lr::AnimatorComponent>()
                                                                    : nullptr;
            },
            ref, "The object's animation player, or None.");

    nb::class_<lr::SceneAssets>(m, "Scene",
                                "A scene loaded with the engine's loaders: objects, meshes and materials, on the CPU. "
                                "Build GPU buffers from it with ResourceRegistry.upload_buffer()/upload_image().")
        .def(nb::init<>())
        .def("save", [](const lr::SceneAssets &assets, const fs::path &path) {
                 lr::SceneSerializer::save(assets, path);
             }, "path"_a,
             "Save this complete scene to a single .lrscene file.")
        .def(
            "load",
            [](lr::SceneAssets &assets, const fs::path &path) -> lr::SceneObject & {
                return assets.scene.getSceneObject(assets.load(path).rootObject);
            },
            "path"_a, ref,
            "Load an OBJ, glTF or GLB file into this scene; returns the new root object it was placed under.")
        .def_prop_ro(
            "objects",
            [](lr::SceneAssets &assets) {
                return liveObjects(assets.scene);
            },
            ref, "Every object, in creation order.")
        .def_prop_ro(
            "roots",
            [](lr::SceneAssets &assets) {
                std::vector<lr::SceneObject *> roots;
                for (lr::SceneObject *object : liveObjects(assets.scene))
                {
                    if (!object->parent())
                    {
                        roots.push_back(object);
                    }
                }
                return roots;
            },
            ref, "Objects without a parent.")
        .def(
            "find",
            [](lr::SceneAssets &assets, const std::string &name) -> lr::SceneObject * {
                for (lr::SceneObject *object : liveObjects(assets.scene))
                {
                    if (object->name == name)
                    {
                        return object;
                    }
                }
                return nullptr;
            },
            "name"_a, ref, "The first object with this name, or None.")
        .def(
            "material",
            [](lr::SceneAssets &assets, uint32_t handle) -> lr::Material & {
                if (handle >= assets.materials.capacity())
                {
                    throw nb::index_error(("material handle " + std::to_string(handle) + " is out of range").c_str());
                }
                return assets.materials.get(handle);
            },
            "handle"_a, ref, "The material a mesh's face_materials entry refers to.")
        .def(
            "update",
            [](lr::SceneAssets &assets, float dt) {
                for (lr::SceneObject *object : liveObjects(assets.scene))
                {
                    if (object->hasComponent<lr::AnimatorComponent>())
                    {
                        object->getComponent<lr::AnimatorComponent>().update(dt);
                    }
                }
            },
            "dt"_a, "Advance every playing animation by dt seconds (moves the animated objects' transforms).")
        .def(
            "add_light",
            [](lr::SceneAssets &assets, const std::string &type, std::array<float, 3> color, float intensity,
               std::array<float, 3> position, std::array<float, 4> rotation, std::array<float, 2> size,
               float innerConeDegrees, float outerConeDegrees, bool twoSided,
               const std::string &name) -> lr::SceneObject & {
                const lr::LightVariant light = makeLight(
                    "add_light", {type, color, intensity, innerConeDegrees, outerConeDegrees, size, twoSided});
                lr::SceneObject &object = assets.scene.createSceneObject();
                object.name             = name;
                auto &transform =
                    object.addComponent<lr::TransformComponent>(glm::vec3(position[0], position[1], position[2]));
                transform.setRotation(glm::quat(rotation[3], rotation[0], rotation[1], rotation[2]));
                object.addComponent<lr::Light>(light);
                return object;
            },
            "type"_a, "color"_a = std::array<float, 3>{1.0f, 1.0f, 1.0f}, "intensity"_a = 1.0f,
            "position"_a = std::array<float, 3>{0.0f, 0.0f, 0.0f},
            "rotation"_a = std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}, "size"_a = std::array<float, 2>{1.0f, 1.0f},
            "inner_cone_degrees"_a = 15.0f, "outer_cone_degrees"_a = 30.0f, "two_sided"_a = true, "name"_a = "Light",
            ref,
            "Add a light object: 'point', 'spot', 'area', 'directional' or 'image' (environment lighting from an Ibl, "
            "scaled by color * intensity). It shines along its rotation's forward axis (spot, area, directional; "
            "area lights shine from both faces unless two_sided=False). "
            "A SceneGpu showing this scene picks it up on the next frame.")
        .def(
            "remove",
            [](lr::SceneAssets &assets, lr::SceneObject &object) {
                if (&object.scene() != &assets.scene)
                {
                    throw std::invalid_argument("remove: the object belongs to a different Scene");
                }
                assets.scene.destroySceneObject(object.id());
            },
            "object"_a,
            "Remove an object and its descendants from the scene. A SceneGpu showing this scene stops drawing "
            "them, and removed lights stop lighting it, from the next frame. The Python objects remain but no "
            "longer appear in objects/roots.");

    m.def(
        "load_scene",
        [](const fs::path &path) {
            std::string extension = path.extension().string();
            std::ranges::transform(extension, extension.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension == ".lrscene")
            {
                return lr::SceneSerializer::load(path);
            }
            auto scene = std::make_unique<lr::SceneAssets>();
            scene->load(path);
            return scene;
        },
        "path"_a, "Load an .lrscene file, OBJ, glTF or GLB file into a new Scene.");
}

// ---------------------------------------------------------------------------------------------
// Engine building blocks: the C++ renderer's scene upload and passes, for Python renderers to
// compose. Their contracts are in docs/python_building_blocks.md.
// ---------------------------------------------------------------------------------------------

// One resource a building block reads or writes, as its passes declared it to the frame graph.
struct ResourceUse
{
    std::string             name;
    std::string             kind;  // "image" or "buffer"
    std::string             usage; // how the block's passes use it, e.g. "sampled", "color_attachment"
    std::optional<VkFormat> format;
};

std::optional<VkFormat> exposedFormat(VkFormat format)
{
    if (format == VK_FORMAT_UNDEFINED)
    {
        return std::nullopt;
    }
    for (const auto &[name, value] : kFormats)
    {
        if (value == format)
        {
            return format;
        }
    }
    return std::nullopt;
}

const char *usageName(lr::ImageUsage usage)
{
    switch (usage)
    {
        case lr::ImageUsage::Sampled:
            return "sampled";
        case lr::ImageUsage::SampledDepth:
            return "sampled_depth";
        case lr::ImageUsage::SampledMultisample:
            return "sampled_multisample";
        case lr::ImageUsage::SampledArray:
            return "sampled_array";
        case lr::ImageUsage::Storage:
            return "storage";
        case lr::ImageUsage::ColorAttachment:
            return "color_attachment";
        case lr::ImageUsage::DepthAttachment:
            return "depth_attachment";
    }
    return "unknown";
}

const char *usageName(lr::BufferUsage usage)
{
    switch (usage)
    {
        case lr::BufferUsage::Uniform:
            return "uniform";
        case lr::BufferUsage::Storage:
            return "storage";
        case lr::BufferUsage::Vertex:
            return "vertex";
        case lr::BufferUsage::Index:
            return "index";
        case lr::BufferUsage::Indirect:
            return "indirect";
    }
    return "unknown";
}

// A building block: the frame-graph passes one engine pass class declared, and what they read and
// write, derived from those declarations and from what the block uploaded to the registry itself.
// Inputs are resources its passes read that something else must provide: read before (or without)
// any of its passes writing them, and not uploaded by the block. Outputs are what its passes write and
// don't consume themselves, plus anything it uploaded without reading — so intermediates between its
// own passes (e.g. HBAO's raw AO before the blur) and its private data (HBAO's params, the LTC
// tables) are neither.
struct PassBlock
{
    std::string              name;
    std::vector<PassHandle>  passes;
    std::vector<std::string> passNames;
    std::vector<ResourceUse> inputs;
    std::vector<ResourceUse> outputs;

    // Call before the block uploads its own resources, so declare() can tell them apart.
    void beginUploads(const ResourceRegistry &registry) { m_namesBeforeUploads = registry.names(); }

    // Runs declarePasses(fg) and records the passes it added and their resources. `alsoOutputs` names
    // results the block's own passes consume that are still meant for others (e.g. Ibl's ibl_env).
    template <typename F>
    void declare(FrameGraph &fg, F &&declarePasses, const std::vector<std::string> &alsoOutputs = {})
    {
        const size_t before = fg.passHandles().size();
        declarePasses(fg);
        const std::vector<PassHandle> all = fg.passHandles();
        passes.assign(all.begin() + static_cast<std::ptrdiff_t>(before), all.end());
        describe(fg, alsoOutputs);
    }

private:
    std::optional<std::vector<std::string>> m_namesBeforeUploads;

    void describe(FrameGraph &fg, const std::vector<std::string> &alsoOutputs)
    {
        const lr::FrameGraphDefinition &definition = fg.definition();
        const ResourceRegistry         &registry   = fg.resources();
        const auto                      contains = [](const std::vector<std::string> &names, const std::string &name) {
            return std::ranges::find(names, name) != names.end();
        };

        std::vector<std::string> uploaded;
        if (m_namesBeforeUploads)
        {
            for (const std::string &resource : registry.names())
            {
                if (!contains(*m_namesBeforeUploads, resource))
                {
                    uploaded.push_back(resource);
                }
            }
            std::ranges::sort(uploaded);
        }

        std::vector<std::string> written; // in first-write order
        std::vector<ResourceUse> writes;
        std::vector<std::string> consumed;
        std::vector<std::string> readNames;
        const auto               read = [&](const ResourceUse &use) {
            readNames.push_back(use.name);
            if (contains(written, use.name))
            {
                consumed.push_back(use.name);
            } else if (!contains(uploaded, use.name) && std::ranges::none_of(inputs, [&](const ResourceUse &u) {
                           return u.name == use.name;
                       }))
            {
                inputs.push_back(use);
            }
        };
        const auto write = [&](const ResourceUse &use) {
            if (!contains(written, use.name))
            {
                written.push_back(use.name);
                writes.push_back(use);
            }
        };

        for (const PassHandle handle : passes)
        {
            const lr::PassDesc &pass = definition.pass(handle);
            passNames.push_back(pass.name);
            for (const lr::ImageUse &use : pass.imageUses)
            {
                const std::string &imageName = definition.name(use.image);
                VkFormat           format    = use.format;
                if (format == VK_FORMAT_UNDEFINED && registry.hasImage(imageName))
                {
                    format = registry.getImage(imageName)->format;
                }
                const ResourceUse info{imageName, "image", usageName(use.usage), exposedFormat(format)};
                const bool        loadsAttachment = use.isAttachment() && use.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
                if (use.access != lr::AccessMode::Write || loadsAttachment)
                {
                    read(info);
                }
                if (use.access != lr::AccessMode::Read)
                {
                    write(info);
                }
            }
            for (const lr::BufferUse &use : pass.bufferUses)
            {
                const ResourceUse info{definition.name(use.buffer), "buffer", usageName(use.usage), std::nullopt};
                if (use.access != lr::AccessMode::Write)
                {
                    read(info);
                }
                if (use.access != lr::AccessMode::Read)
                {
                    write(info);
                }
            }
        }
        for (const ResourceUse &use : writes)
        {
            if (!contains(consumed, use.name) || contains(alsoOutputs, use.name))
            {
                outputs.push_back(use);
            }
        }
        for (const std::string &resource : uploaded)
        {
            if (!contains(readNames, resource) && !contains(written, resource))
            {
                const bool image  = registry.hasImage(resource);
                const auto format = image ? exposedFormat(registry.getImage(resource)->format) : std::nullopt;
                outputs.push_back({resource, image ? "image" : "buffer", "uploaded", format});
            }
        }
    }
};

struct GeometryBlock : PassBlock
{
    std::unique_ptr<lr::GeometryPass> pass;
    lr::CallbackConnection            geometryRebuilt; // refreshes the draw lists
};

struct AmbientOcclusionBlock : PassBlock
{
    std::unique_ptr<lr::AmbientOcclusionPass> pass;
    ResourceRegistry                         *registry = nullptr;

    // Field setter that pushes the change to the GPU params buffer.
    template <typename T> void set(T lr::AmbientOcclusionPass::Config::*field, T value)
    {
        pass->config().*field = value;
        pass->updateParams(*registry);
    }
};

struct PbrBlock : PassBlock
{
    std::unique_ptr<lr::PbrPass> pass;
    lr::CallbackConnection       lightsUploaded; // keeps the light count current
};

struct CompositeBlock : PassBlock
{
    std::unique_ptr<lr::CompositePass> pass;
};

struct IblBlock : PassBlock
{
    lr::IBLPass::Config config;
};

std::string describeBlock(const PassBlock &block)
{
    const auto list = [](const std::vector<ResourceUse> &uses) {
        std::string text;
        for (const ResourceUse &use : uses)
        {
            text += "\n    " + use.name + " (" + use.kind + ", " + use.usage + ")";
        }
        return text.empty() ? std::string("\n    (none)") : text;
    };
    return block.name + "\n  inputs:" + list(block.inputs) + "\n  outputs:" + list(block.outputs);
}

void bindBuildingBlocks(nb::module_ &m)
{
    nb::class_<ResourceUse>(m, "ResourceUse", "A resource a building block reads or writes.")
        .def_ro("name", &ResourceUse::name, "Its name in the frame graph / ResourceRegistry.")
        .def_ro("kind", &ResourceUse::kind, "'image' or 'buffer'.")
        .def_ro("usage", &ResourceUse::usage,
                "How the block uses it: images 'sampled', 'sampled_depth', 'sampled_array', 'storage', "
                "'color_attachment' or 'depth_attachment'; buffers 'uniform', 'storage', 'vertex', 'index' or "
                "'indirect'.")
        .def_ro("format", &ResourceUse::format, "Image format, when known; None for buffers.")
        .def("__repr__", [](const ResourceUse &use) {
            return "ResourceUse('" + use.name + "', " + use.kind + ", " + use.usage + ")";
        });

    nb::class_<PassBlock>(m, "EnginePass",
                          "Base class of the engine's passes (GeometryPass, AmbientOcclusionPass, PbrPass, "
                          "CompositePass) and of Ibl. Constructing one declares its passes in the viewer's frame "
                          "graph; inputs/outputs list what they read and write, taken from those declarations.")
        .def_ro("name", &PassBlock::name)
        .def_ro("passes", &PassBlock::passes, "Handles of the frame-graph passes it declared, e.g. for depends_on().")
        .def_ro("pass_names", &PassBlock::passNames)
        .def_ro("inputs", &PassBlock::inputs,
                "Resources it reads that something else must provide (another block, or your own pass).")
        .def_ro("outputs", &PassBlock::outputs, "Resources it writes, for later passes to read.")
        .def("describe", &describeBlock, "Inputs and outputs as readable text.")
        .def("__repr__", &describeBlock);

    nb::class_<lr::SceneGpu>(m, "SceneGpu",
                             "A Scene's GPU buffers in the engine's layout — what the engine's passes read: camera "
                             "UBO, lights, mesh vertex/index buffers, materials and textures, skins. Kept in sync "
                             "every frame (camera moves, animations, skinning). One per Viewer.")
        .def(
            "__init__",
            [](lr::SceneGpu *self, Viewer &viewer, lr::SceneAssets &assets, OrbitCamera &camera) {
                std::vector<lr::SceneObject *> meshObjects;
                for (lr::SceneObject *object : liveObjects(assets.scene))
                {
                    if (object->hasComponent<lr::MeshComponent>() && object->hasComponent<lr::TransformComponent>())
                    {
                        meshObjects.push_back(object);
                    }
                }
                if (meshObjects.empty())
                {
                    throw std::invalid_argument("SceneGpu: the scene has no meshes (load one first)");
                }
                if (viewer.resources().hasBuffer("meshPositionBuffer"))
                {
                    throw std::logic_error("SceneGpu: this Viewer already has one");
                }

                new (self) lr::SceneGpu(viewer.resources(), assets.scene, assets.meshes, assets.materials);
                for (lr::SceneObject *object : meshObjects)
                {
                    self->addMeshObject(*object);
                }
                self->setCamera(camera.object());
                const VkExtent2D extent = viewer.resources().getExtent();
                self->setAspect(
                    extent.height == 0 ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height));
                self->initialize();
                self->registerCallbacks(viewer);
                holdWhileViewerRuns(viewer, self);
            },
            "viewer"_a, "scene"_a, "camera"_a, nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(),
            "Upload `scene` (its meshes and lights as they are now) and keep it in sync with `camera`. Every light "
            "also gets a quad mesh, drawn with the scene (bright for area lights, invisible otherwise). The quads "
            "belong to the SceneGpu: `scene` itself isn't modified and can be shown again later.")
        .def_prop_ro("camera_buffer", &lr::SceneGpu::cameraBufferName,
                     "Camera UBO: mat4 view, proj, viewProj, invView, invProj; vec4 position (std140, 336 bytes).")
        .def_prop_ro("light_buffer", &lr::SceneGpu::lightBufferName, "Light SSBO, as pbr.frag reads it.")
        .def_prop_ro("num_lights", &lr::SceneGpu::numLights, "Lights currently in the light buffer.")
        .def_prop_ro("max_lights", &lr::SceneGpu::maxLights,
                     "Most lights the light buffer holds; more raises an error on the next frame.")
        .def_prop_ro("mesh_count",
                     [](const lr::SceneGpu &gpu) {
                         return gpu.meshObjects().size();
                     })
        .def_prop_ro(
            "buffers",
            [](const lr::SceneGpu &gpu) {
                nb::dict buffers;
                buffers["camera"]         = gpu.cameraBufferName();
                buffers["lights"]         = gpu.lightBufferName();
                buffers["positions"]      = gpu.meshPositionBufferName();
                buffers["attributes"]     = gpu.meshVertexBufferName();
                buffers["indices"]        = gpu.meshIndexBufferName();
                buffers["face_groups"]    = gpu.meshFaceGroupBufferName();
                buffers["materials"]      = gpu.materialUploadResult().materialInfoBufferName;
                buffers["joint_matrices"] = gpu.skinJointMatricesBufferName();
                return buffers;
            },
            "Names of the buffers it keeps, by role, for your own passes to bind.");

    nb::class_<IblBlock, PassBlock>(m, "Ibl",
                                    "Image-based lighting, precomputed once from an HDR environment (compute passes "
                                    "run immediately, in their own frame graph). PbrPass lights the scene with it; "
                                    "CompositePass draws its environment as the sky.")
        .def(
            "__init__",
            [](IblBlock *self, Viewer &viewer, std::optional<fs::path> hdri, uint32_t envRes, uint32_t irrRes,
               uint32_t pfRes, uint32_t pfMips) {
                new (self) IblBlock();
                self->name   = "Ibl";
                self->config = {.hdriPath = hdri.value_or(fs::path()),
                                .envRes   = envRes,
                                .irrRes   = irrRes,
                                .pfRes    = pfRes,
                                .pfMips   = pfMips};
                lr::IBLPass pass(self->config);
                self->beginUploads(viewer.resources());
                pass.uploadResources(viewer.resources());
                FrameGraph graph(viewer.context(), viewer.resources());
                // ibl_env feeds the irradiance/prefilter passes, and is also the sky CompositePass draws.
                self->declare(graph,
                              [&](FrameGraph &fg) {
                                  pass.build(fg);
                              },
                              {"ibl_env"});
                nb::gil_scoped_release release; // the precompute waits for the GPU
                graph.executeAndWait({
                    {"ibl_irradiance", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {"ibl_prefiltered", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                });
            },
            "viewer"_a, "hdri"_a = nb::none(), "env_res"_a = 2048, "irr_res"_a = 32, "pf_res"_a = 2048, "pf_mips"_a = 8,
            "hdri: an equirectangular .hdr image (None: a black environment). env_res/pf_res: cubemap face sizes; "
            "irr_res: irradiance face size; pf_mips: prefiltered roughness levels.")
        .def_prop_ro("pf_mips", [](const IblBlock &ibl) {
            return ibl.config.pfMips;
        });

    nb::class_<GeometryBlock, PassBlock>(m, "GeometryPass",
                                         "The engine's G-buffer pass: draws every SceneGpu mesh with its material. "
                                         "See docs/python_building_blocks.md for the G-buffer encoding.")
        .def(
            "__init__",
            [](GeometryBlock *self, Viewer &viewer, lr::SceneGpu &gpu) {
                new (self) GeometryBlock();
                self->name = "GeometryPass";
                self->pass = std::make_unique<lr::GeometryPass>(gpu.geometryPassConfig());
                self->declare(viewer.frameGraph(), [&](FrameGraph &fg) {
                    self->pass->build(fg, lr::conventions::geometryMeshLayout());
                });
                self->geometryRebuilt = gpu.onGeometryRebuilt([pass = self->pass.get()](const lr::SceneGpu &rebuilt) {
                    pass->setSceneGeometry(rebuilt.meshPositions(), rebuilt.indexBuffer(), rebuilt.meshTransforms(),
                                           rebuilt.geometryObjects(), rebuilt.skinUploadResult().drawInfos);
                });
                holdWhileViewerRuns(viewer, self); // the pass's execute callback reads self->pass
            },
            "viewer"_a, "scene_gpu"_a, nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
        .def_prop_rw(
            "skinning",
            [](const GeometryBlock &block) {
                return block.pass->isSkinningEnabled();
            },
            [](GeometryBlock &block, bool enabled) {
                block.pass->setSkinningEnabled(enabled);
            },
            "Draw skinned meshes posed (True, default) or in their rest pose.");

    using AoConfig        = lr::AmbientOcclusionPass::Config;
    const auto aoProperty = [](float AoConfig::*field) {
        return [field](AmbientOcclusionBlock &block, float value) {
            block.set(field, value);
        };
    };
    const auto aoIntProperty = [](int AoConfig::*field) {
        return [field](AmbientOcclusionBlock &block, int value) {
            block.set(field, value);
        };
    };
    nb::class_<AmbientOcclusionBlock, PassBlock>(m, "AmbientOcclusionPass",
                                                 "Horizon-based ambient occlusion (HBAO) from the G-buffer depth, "
                                                 "then a bilateral blur. Parameters can change while running.")
        .def(
            "__init__",
            [](AmbientOcclusionBlock *self, Viewer &viewer, lr::SceneGpu &gpu, float sphereRadius, int numSteps,
               int numDirs, float tanAngleBias, float aoScalar) {
                new (self) AmbientOcclusionBlock();
                self->name     = "AmbientOcclusionPass";
                self->registry = &viewer.resources();
                self->pass     = std::make_unique<lr::AmbientOcclusionPass>(AoConfig{
                    .cameraBufferResourceName = gpu.cameraBufferName(),
                    .sphereRadius             = sphereRadius,
                    .numSteps                 = numSteps,
                    .numDirs                  = numDirs,
                    .tanAngleBias             = tanAngleBias,
                    .aoScalar                 = aoScalar,
                });
                self->beginUploads(viewer.resources());
                self->pass->uploadResources(viewer.resources());
                self->declare(viewer.frameGraph(), [&](FrameGraph &fg) {
                    self->pass->build(fg);
                });
            },
            "viewer"_a, "scene_gpu"_a, "sphere_radius"_a = 0.5f, "num_steps"_a = 16, "num_dirs"_a = 8,
            "tan_angle_bias"_a = 0.364f, "ao_scalar"_a = 2.0f, nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
            "sphere_radius: world-space sampling radius (scale it to the model); num_steps/num_dirs: samples per "
            "direction / directions; tan_angle_bias: ignores horizons below this slope; ao_scalar: strength.")
        .def_prop_rw(
            "sphere_radius",
            [](const AmbientOcclusionBlock &b) {
                return b.pass->config().sphereRadius;
            },
            aoProperty(&AoConfig::sphereRadius))
        .def_prop_rw(
            "num_steps",
            [](const AmbientOcclusionBlock &b) {
                return b.pass->config().numSteps;
            },
            aoIntProperty(&AoConfig::numSteps))
        .def_prop_rw(
            "num_dirs",
            [](const AmbientOcclusionBlock &b) {
                return b.pass->config().numDirs;
            },
            aoIntProperty(&AoConfig::numDirs))
        .def_prop_rw(
            "tan_angle_bias",
            [](const AmbientOcclusionBlock &b) {
                return b.pass->config().tanAngleBias;
            },
            aoProperty(&AoConfig::tanAngleBias))
        .def_prop_rw(
            "ao_scalar",
            [](const AmbientOcclusionBlock &b) {
                return b.pass->config().aoScalar;
            },
            aoProperty(&AoConfig::aoScalar));

    nb::class_<PbrBlock, PassBlock>(m, "PbrPass",
                                    "The engine's deferred lighting: Cook-Torrance BRDF for the SceneGpu's lights "
                                    "(area lights via LTC). The Ibl lights the scene through 'image' lights, darkened "
                                    "by hbao_ao.")
        .def(
            "__init__",
            [](PbrBlock *self, Viewer &viewer, lr::SceneGpu &gpu, const IblBlock &ibl) {
                new (self) PbrBlock();
                self->name = "PbrPass";
                self->pass = std::make_unique<lr::PbrPass>(lr::PbrPass::Config{
                    .cameraBufferResourceName = gpu.cameraBufferName(),
                    .lightBufferResourceName  = gpu.lightBufferName(),
                    .numLights                = gpu.numLights(),
                    .pfMips                   = ibl.config.pfMips,
                });
                self->beginUploads(viewer.resources());
                self->pass->uploadResources(viewer.resources());
                self->declare(viewer.frameGraph(), [&](FrameGraph &fg) {
                    self->pass->build(fg);
                });
                self->lightsUploaded = gpu.onLightsUploaded([pass = self->pass.get()](uint32_t numLights) {
                    pass->setNumLights(numLights);
                });
                holdWhileViewerRuns(viewer, self); // the pass's execute callback reads the light count from it
            },
            "viewer"_a, "scene_gpu"_a, "ibl"_a, nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>());

    nb::class_<CompositeBlock, PassBlock>(m, "CompositePass",
                                          "Tone maps an HDR image (Reinhard) to the window, with the Ibl environment "
                                          "as the sky wherever GeometryPass drew nothing.")
        .def(
            "__init__",
            [](CompositeBlock *self, Viewer &viewer, lr::SceneGpu &gpu, const std::string &input,
               const std::string &output, std::optional<VkFormat> outputFormat, const std::string &name) {
                new (self) CompositeBlock();
                self->name = "CompositePass";
                self->pass = std::make_unique<lr::CompositePass>(lr::CompositePass::Config{
                    .cameraBufferResourceName = gpu.cameraBufferName(),
                    .outputFormat             = outputFormat.value_or(viewer.swapchainFormat()),
                    .inputImage               = input,
                    .outputImage              = output,
                    .passName                 = name,
                });
                self->declare(viewer.frameGraph(), [&](FrameGraph &fg) {
                    self->pass->build(fg);
                });
            },
            "viewer"_a, "scene_gpu"_a, "input"_a = "pbr", "output"_a = "swapchain", "output_format"_a = nb::none(),
            "name"_a = "composite", nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
            "input: the HDR image to show (PbrPass's 'pbr', or your own pass's output). output/output_format: where "
            "to write (default: the window).");
}

enum class FrameCount
{
    One,
    UntilClosed,
};

// Runs one frame (step) or frames until the window closes (run), with the Python-side bookkeeping
// around them: callback errors are trapped during the frames and re-raised afterwards, and once the
// window has closed nothing can call into Python again, so the held callbacks are released. Returns
// whether the window is still open.
bool driveFrames(Viewer &v, FrameCount count)
{
    // The ImGui pass composites onto the swapchain, so it must come after every user pass.
    if (!v.hasImguiPass())
    {
        v.addImguiPass();
    }
    // Validation errors left over from a different (e.g. already destroyed) Viewer don't belong to it.
    if (g_callbackErrors.validationSource != &v)
    {
        g_callbackErrors.clearValidationErrors();
    }
    g_callbackErrors.running = &v;
    bool open                = false;
    try
    {
        if (count == FrameCount::One)
        {
            open = v.step();
        } else
        {
            v.run();
        }
    } catch (...)
    {
        g_callbackErrors.running = nullptr;
        g_callbackErrors.error.reset();
        g_callbackErrors.clearValidationErrors();
        g_callbackSlots.releaseAll();
        throw;
    }
    g_callbackErrors.running = nullptr;
    if (!open)
    {
        g_callbackSlots.releaseAll();
    }
    // A trapped error asked the window to close, so the frame that raised it was the last.
    g_callbackErrors.rethrowPending(v);
    return open;
}

void bindViewer(nb::module_ &m)
{
    nb::class_<lr::CallbackConnection>(m, "CallbackConnection",
                                       "A scoped callback registration. Keep it alive while the callback is needed.")
        .def("disconnect", &lr::CallbackConnection::disconnect);

    nb::class_<Viewer>(m, "Viewer",
                       "Window + Vulkan device + frame graph. Declare passes, then call run(), or call step() "
                       "in your own loop.")
        .def(
            "__init__",
            [](Viewer *self, const std::string &title, int width, int height, bool validation,
               bool raiseValidationErrors, bool gui) {
                if (g_liveViewers > 0)
                {
                    // An earlier Viewer that never ran (e.g. an exception before run()) is kept alive only
                    // by the reference cycles through its callbacks, SceneGpu and passes, which run()
                    // would have released: release them now, and collect it.
                    g_callbackSlots.releaseAll();
                    nb::module_::import_("gc").attr("collect")();
                    if (g_liveViewers > 0)
                    {
                        throw std::runtime_error(
                            "lr supports one Viewer at a time, and an earlier one is still referenced: delete it "
                            "(and anything holding it, e.g. `del viewer`) before creating another");
                    }
                }
                new (self) Viewer(Viewer::Config{
                    .title            = title,
                    .width            = width,
                    .height           = height,
                    .enableValidation = validation,
                    .enableGui        = gui,
                });
                self->ownConnection(self->onUpdate([token = std::make_shared<ViewerToken>()](float, VkExtent2D) {}));
                // Other Python threads run while a frame waits on the GPU or the display (fences, vsync,
                // present). Nothing inside these calls touches Python; callbacks and recording keep the GIL.
                self->setBlockingCallWrapper([](const std::function<void()> &call) {
                    nb::gil_scoped_release release;
                    call();
                });
                if (validation && raiseValidationErrors)
                {
                    self->onValidationError([self](std::string_view message) {
                        g_callbackErrors.recordValidationError(*self, message);
                    });
                }
            },
            "title"_a = "lr", "width"_a = 1600, "height"_a = 900, "validation"_a = true,
            "raise_validation_errors"_a = true, "gui"_a = true,
            "With validation on (the default), a validation-layer error closes the window and run() raises "
            "VulkanValidationError; pass raise_validation_errors=False to only log them. gui=False creates no "
            "ImGui context: on_gui() raises and nothing is drawn over your frame.")
        .def_prop_ro("gui", &Viewer::guiEnabled, "Whether this Viewer has ImGui (see the gui argument).")
        .def_prop_ro("frame_graph", &Viewer::frameGraph, nb::rv_policy::reference_internal)
        .def_prop_ro("resources", &Viewer::resources, nb::rv_policy::reference_internal)
        .def_prop_ro("swapchain_format", &Viewer::swapchainFormat)
        .def(
            "on_update",
            [](Viewer &v, nb::callable callback) {
                return v.onUpdate([slot = g_callbackSlots.hold(std::move(callback))](float dt, VkExtent2D extent) {
                    if (*slot)
                    {
                        g_callbackErrors.guard([&] {
                            (*slot)(dt, toTuple(extent));
                        });
                    }
                });
            },
            "callback"_a,
            "Register callback(dt, extent) before rendering. Keep the returned connection alive.")
        .def(
            "on_late_update",
            [](Viewer &v, nb::callable callback) {
                return v.onLateUpdate([slot = g_callbackSlots.hold(std::move(callback))](float dt, VkExtent2D extent) {
                    if (*slot)
                    {
                        g_callbackErrors.guard([&] {
                            (*slot)(dt, toTuple(extent));
                        });
                    }
                });
            },
            "callback"_a, "Like on_update, but after every on_update callback. Keep the connection alive.")
        .def(
            "on_gui",
            [](Viewer &v, nb::callable callback) {
                if (!v.guiEnabled())
                {
                    throw std::logic_error("on_gui: this Viewer was created with gui=False");
                }
                return v.onGui([slot = g_callbackSlots.hold(std::move(callback))] {
                    if (!*slot)
                    {
                        return;
                    }
                    g_gui.active      = true;
                    g_gui.openWindows = 0;
                    g_callbackErrors.guard([&] {
                        (*slot)();
                    });
                    // Close what the callback left open, so ImGui's frame stays balanced.
                    for (; g_gui.openWindows > 0; --g_gui.openWindows)
                    {
                        ImGui::End();
                    }
                    g_gui.active = false;
                });
            },
            "callback"_a,
            "callback(), called every frame to build ImGui panels with lr.gui (e.g. `with lr.gui.window(...)`).")
        .def_prop_ro("input", &Viewer::input, nb::rv_policy::reference_internal,
                     "Keyboard and mouse state for this window.")
        .def(
            "run",
            [](Viewer &v) {
                driveFrames(v, FrameCount::UntilClosed);
            },
            "Compile the frame graph and run frames until the window closes. An exception raised in any "
            "callback closes the window and is re-raised here. Once the window has closed, its callbacks are "
            "released: a Viewer runs once.")
        .def(
            "step",
            [](Viewer &v) {
                return driveFrames(v, FrameCount::One);
            },
            "Render one frame (compiling the frame graph on the first call) and return True, or False once the "
            "window has closed: `while viewer.step(): ...` lets your code own the loop. The window only responds "
            "while it is being stepped. Callbacks run as with run(); an exception raised in one closes the window "
            "and is re-raised from this step. Once the window has closed, its callbacks are released.")
        .def_prop_ro("is_open", &Viewer::isOpen, "True until the window has closed.")
        .def("close", &Viewer::requestClose,
             "Close the window after the current frame (run() returns, step() "
             "returns False).");
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
    // Plain assert() (e.g. ImGui's IM_ASSERT) reports through _wassert, which bypasses the hook above:
    // send its message to stderr, and route the abort() that follows through the interpreter's runtime.
    _set_error_mode(_OUT_TO_STDERR);
    std::signal(SIGABRT, [](int) {
        abortThroughInterpreterRuntime();
    });
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
            viewer.ownConnection(viewer.onLateUpdate([frames = std::make_shared<int>(0), afterFrames](float, VkExtent2D) {
                if (++*frames > afterFrames)
                {
                    throw std::runtime_error("lr._testing: injected C++ failure inside the frame loop");
                }
            }));
        },
        "viewer"_a, "after_frames"_a);

    // Synthetic input, as if from the window: seen by viewer.input on the next frame.
    testing.def(
        "inject_mouse_move",
        [](Viewer &viewer, double x, double y) {
            viewer.input().notifyMouseMove(x, y);
        },
        "viewer"_a, "x"_a, "y"_a);
    testing.def(
        "inject_mouse_button",
        [](Viewer &viewer, MouseButton button, bool pressed) {
            viewer.input().notifyMouseButton(static_cast<int>(button), pressed ? GLFW_PRESS : GLFW_RELEASE);
        },
        "viewer"_a, "button"_a, "pressed"_a);
    testing.def(
        "inject_scroll",
        [](Viewer &viewer, double delta) {
            viewer.input().notifyScroll(delta);
        },
        "viewer"_a, "delta"_a);
    testing.def(
        "inject_key",
        [](Viewer &viewer, Key key, bool pressed) {
            viewer.input().notifyKey(static_cast<int>(key), pressed ? GLFW_PRESS : GLFW_RELEASE);
        },
        "viewer"_a, "key"_a, "pressed"_a);

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
            } else if (kind == "assert")
            {
                // A plain assert(), as third-party code like ImGui uses (compiled out in Release).
                volatile bool ok = false;
                assert(ok && "lr._testing: plain assert");
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
    bindInput(m);
    bindGui(m);
    bindViewer(m);
    bindCamera(m);
    // lr.engine: the engine's own, opinionated renderer (its scene model and loaders, its GPU layout,
    // its passes). Everything else in lr is the general-purpose frame graph it's built on.
    nb::module_ engine = m.def_submodule(
        "engine", "The engine's preassembled renderer: scenes, their GPU layout, IBL and the deferred passes.");
    bindScene(engine);
    bindBuildingBlocks(engine);
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

    m.attr("SWAPCHAIN")  = Viewer::kBackbufferName;
    m.attr("SHADER_DIR") = lr::paths::shaderDir;
    m.attr("ASSET_DIR")  = lr::paths::assetDir;

    // Covers scripts that never reach run() (or exit mid-way): drop the callbacks while the
    // interpreter can still destroy the Viewers they keep alive.
    nb::module_::import_("atexit").attr("register")(nb::cpp_function([] {
        g_callbackSlots.releaseAll();
    }));
}
