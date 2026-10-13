#include "SceneDocument.hpp"
#include "EditorSession.hpp"

#include "core/Paths.hpp"
#include "core/loaders/ObjExporter.hpp"
#include "core/app/Viewer.hpp"
#include "core/editor/ErrorReporter.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/SceneManager.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"
#include "features/fluid/FluidComponent.hpp"
#include "features/fluid/import/BlenderFluidCacheImporter.hpp"
#include "features/linear_blend_skinning/SkinComponent.hpp"
#include "features/rigid_body/PhysicsWorld.hpp"

#include <ImGuiFileDialog.h>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include <exception>
#include <stdexcept>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace lr
{

namespace fs = std::filesystem;

namespace
{
// Dialog keys. Shared between the open and display calls, so they live here rather than being
// spelled out at each site.
constexpr const char *kLoadDialog   = "LoadNativeScene";
constexpr const char *kSaveDialog   = "SaveNativeScene";
constexpr const char *kImportDialog = "ImportScene";
constexpr const char *kExportDialog = "ExportGeometry";
constexpr const char *kFluidCacheDialog = "ImportFluidCache";

constexpr const char *kSceneExtension = ".lrscene";
constexpr const char *kObjExtension   = ".obj";

constexpr ImVec2 kDialogSize(640.0f, 360.0f);

ImGuiFileDialogFlags commonDialogFlags()
{
    return ImGuiFileDialogFlags_Modal | ImGuiFileDialogFlags_CaseInsensitiveExtentionFiltering |
           ImGuiFileDialogFlags_ShowDevicesButton;
}
} // namespace

SceneDocument::SceneDocument(Viewer &viewer, SceneManager &sceneManager, EditorSession &editor,
                             PhysicsWorld &physicsWorld, SceneLoaderConfig importConfig, ErrorReporter &errors)
    : m_viewer(viewer), m_sceneManager(sceneManager), m_editor(editor), m_physicsWorld(physicsWorld),
      m_errors(errors), m_importConfig(std::move(importConfig))
{}

void SceneDocument::openLoadDialog()
{
    IGFD::FileDialogConfig dialogConfig;
    dialogConfig.path  = m_path ? m_path->parent_path().string() : ".";
    dialogConfig.flags = commonDialogFlags();
    ImGuiFileDialog::Instance()->OpenDialog(kLoadDialog, "Load Scene", kSceneExtension, dialogConfig);
}

void SceneDocument::openSaveAsDialog()
{
    IGFD::FileDialogConfig dialogConfig;
    dialogConfig.path     = m_path ? m_path->parent_path().string() : ".";
    dialogConfig.fileName = m_path ? m_path->filename().string() : std::string("scene") + kSceneExtension;
    dialogConfig.flags    = commonDialogFlags() | ImGuiFileDialogFlags_ConfirmOverwrite;
    ImGuiFileDialog::Instance()->OpenDialog(kSaveDialog, "Save Scene As", kSceneExtension, dialogConfig);
}

void SceneDocument::openImportDialog()
{
    IGFD::FileDialogConfig dialogConfig;
    dialogConfig.path  = ".";
    dialogConfig.flags = commonDialogFlags() | ImGuiFileDialogFlags_ReadOnlyFileNameField;
    ImGuiFileDialog::Instance()->OpenDialog(kImportDialog, "Import Mesh or Animated Mesh", ".obj,.gltf,.glb",
                                            dialogConfig);
}

void SceneDocument::openImportFluidCacheDialog()
{
    IGFD::FileDialogConfig dialogConfig;
    dialogConfig.path  = (paths::assetDir / "samples" / "fluid").string();
    dialogConfig.flags = commonDialogFlags();
    // A null filter is ImGuiFileDialog's directory-selection mode: a baked cache is a directory of
    // per-frame files, so there is no single file to pick.
    ImGuiFileDialog::Instance()->OpenDialog(kFluidCacheDialog, "Import Baked Fluid Cache (select folder)", nullptr,
                                            dialogConfig);
}

bool SceneDocument::canExportGeometry()
{
    // The registered mesh objects are exactly what the renderer draws, so asking them is what makes
    // the menu item agree with what is on screen. Empty before anything is imported.
    return !collectExportObjects().empty();
}

void SceneDocument::openExportGeometryDialog()
{
    if (!canExportGeometry())
    {
        return;
    }

    IGFD::FileDialogConfig dialogConfig;
    dialogConfig.path = m_path ? m_path->parent_path().string() : ".";
    // Named after the scene rather than after any one object, since the whole scene is written.
    dialogConfig.fileName = (m_path ? m_path->stem().string() : std::string("scene")) + kObjExtension;
    dialogConfig.flags    = commonDialogFlags() | ImGuiFileDialogFlags_ConfirmOverwrite;
    ImGuiFileDialog::Instance()->OpenDialog(kExportDialog, "Export Scene Geometry", kObjExtension, dialogConfig);
}

std::vector<ObjExportObject> SceneDocument::collectExportObjects() const
{
    std::vector<ObjExportObject> objects;
    for (SceneObject *object : m_sceneManager.gpu().meshObjects())
    {
        // Registration already required both components (SceneGpu::addMeshObject), but a null entry
        // or a component removed since would otherwise reach the exporter as a null mesh.
        if (!object)
        {
            continue;
        }
        const MeshComponent *mesh = object->findComponent<MeshComponent>();
        if (!mesh)
        {
            continue;
        }

        // A posed character should export posed, so a skinned object hands the exporter its
        // evaluated joint palette. Skinning switched off is deliberately treated the same as no
        // skin at all -- bind-pose positions through the object's transform -- because that is what
        // the viewport is drawing in that state (see SkinComponent's own note).
        std::span<const glm::mat4> jointMatrices;
        if (const SkinComponent *skin = object->findComponent<SkinComponent>(); skin && skin->skin().skinningEnabled())
        {
            // Already evaluated for this frame by the animation/skin update, the same matrices the
            // geometry pass is drawing with; re-evaluating here would both duplicate that work and
            // risk disagreeing with what is on screen.
            jointMatrices = skin->skin().jointMatrices();
        }

        objects.push_back({&mesh->mesh(), object->worldMatrix(), object->name, jointMatrices});
    }
    return objects;
}

void SceneDocument::importPrimitive(const std::string &fileName, const std::string &objectName)
{
    try
    {
        const fs::path primitivePath = paths::assetDir / "runtime" / "primitives" / fileName;
        m_viewer.context().waitIdle();
        SceneObject &root = m_sceneManager.load(primitivePath, m_importConfig);
        // The loader names the root after the file; the menu offered a shape, so the hierarchy says
        // the shape. Named unconditionally because every primitive file holds exactly one object.
        root.name = objectName;
        resettleAfterSceneChange();
        m_importError.clear();
    } catch (const std::exception &e)
    {
        m_importError = e.what();
        m_errors.report("Could not add primitive", e.what());
        spdlog::error("Failed to add primitive '{}': {}", objectName, e.what());
    }
}

void SceneDocument::importFluidCache(const fs::path &cacheDirectory)
{
    try
    {
        OfflineFluidSimulation clip = BlenderFluidCacheImporter::importDirectory(cacheDirectory);
        const std::string      name = clip.name();
        const uint32_t         particles = clip.maximumParticleCount();
        const std::size_t      frames    = clip.frames().size();

        // No GPU resource is bound against the clip store, so unlike a geometry import this does
        // not need a device idle or a geometry rebuild -- the fluid runtime picks the clip up when
        // the new object is selected.
        const OfflineFluidSimulationHandle handle = m_sceneManager.offlineFluids().add(std::move(clip));

        SceneObject &object = m_sceneManager.scene().createSceneObject();
        object.name         = name.empty() ? "Baked Fluid" : name;
        object.addComponent<TransformComponent>();

        FluidComponent::BakedConfig baked;
        baked.simulation = handle;
        object.addComponent<FluidComponent>(baked);

        // The fluid runtime follows the selected object carrying a FluidComponent, so selecting the
        // import is what starts it playing rather than leaving the user to find it in the hierarchy.
        m_sceneManager.scene().selectObject(object.id());

        m_editor.onSceneContentChanged();
        m_importError.clear();
        spdlog::info("Imported baked fluid cache '{}': {} frames, {} particles per frame", object.name, frames,
                     particles);
    } catch (const std::exception &e)
    {
        m_importError = e.what();
        m_errors.report("Could not import fluid cache", e.what());
        spdlog::error("Failed to import fluid cache '{}': {}", cacheDirectory.string(), e.what());
    }
}

void SceneDocument::save()
{
    if (!m_path)
    {
        return;
    }

    try
    {
        m_sceneManager.save(*m_path);
        m_persistenceError.clear();
    } catch (const std::exception &e)
    {
        m_persistenceError = e.what();
        m_errors.report("Could not save scene", e.what());
        spdlog::error("Failed to save scene: {}", e.what());
    }
}

void SceneDocument::resettleAfterSceneChange()
{
    m_sceneManager.rebuildGeometry(); // refreshes geometryPass via onGeometryRebuilt
    m_editor.onSceneContentChanged();
    m_physicsWorld.onSceneChanged();
    m_viewer.frameGraph().compile();
}

void SceneDocument::drawDialogs()
{
    if (ImGuiFileDialog::Instance()->Display(kLoadDialog, ImGuiWindowFlags_NoCollapse, kDialogSize))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            try
            {
                const fs::path scenePath(ImGuiFileDialog::Instance()->GetFilePathName());
                m_viewer.context().waitIdle();
                m_sceneManager.loadScene(scenePath);
                resettleAfterSceneChange();
                m_path = scenePath;
                m_persistenceError.clear();
                // After the scene itself is settled: the callback rebuilds state derived from the
                // document, which must see the loaded contents rather than the previous scene's.
                if (m_onSceneLoaded)
                {
                    m_onSceneLoaded();
                }
            } catch (const std::exception &e)
            {
                m_persistenceError = e.what();
                m_errors.report("Could not load scene", e.what());
                spdlog::error("Failed to load scene: {}", e.what());
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display(kSaveDialog, ImGuiWindowFlags_NoCollapse, kDialogSize))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            try
            {
                fs::path selectedPath(ImGuiFileDialog::Instance()->GetFilePathName());
                if (selectedPath.extension() != kSceneExtension)
                {
                    selectedPath += kSceneExtension;
                }
                m_sceneManager.save(selectedPath);
                m_path = std::move(selectedPath);
                m_persistenceError.clear();
            } catch (const std::exception &e)
            {
                m_persistenceError = e.what();
                m_errors.report("Could not save scene", e.what());
                spdlog::error("Failed to save scene: {}", e.what());
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display(kExportDialog, ImGuiWindowFlags_NoCollapse, kDialogSize))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            try
            {
                // Re-collected rather than captured at open time: the dialog is modeless as far as
                // the scene is concerned, so objects can be added, moved or deleted between opening
                // it and confirming it, and the export should reflect the scene as it is now.
                const std::vector<ObjExportObject> objects = collectExportObjects();
                if (objects.empty())
                {
                    throw std::runtime_error("The scene has no renderable geometry to export.");
                }

                fs::path selectedPath(ImGuiFileDialog::Instance()->GetFilePathName());
                if (selectedPath.extension() != kObjExtension)
                {
                    selectedPath += kObjExtension;
                }
                ObjExporter::write(objects, selectedPath);
                m_persistenceError.clear();
            } catch (const std::exception &e)
            {
                m_persistenceError = e.what();
                m_errors.report("Could not export geometry", e.what());
                spdlog::error("Failed to export geometry: {}", e.what());
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display(kImportDialog, ImGuiWindowFlags_NoCollapse, kDialogSize))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            try
            {
                const fs::path selectedPath(ImGuiFileDialog::Instance()->GetFilePathName());
                m_viewer.context().waitIdle();
                m_sceneManager.load(selectedPath, m_importConfig);
                resettleAfterSceneChange();
                m_importError.clear();
            } catch (const std::exception &e)
            {
                m_importError = e.what();
                m_errors.report("Could not import asset", e.what());
                spdlog::error("Failed to import scene: {}", e.what());
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display(kFluidCacheDialog, ImGuiWindowFlags_NoCollapse, kDialogSize))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            // Directory mode reports the chosen folder through GetCurrentPath; there is no file name.
            importFluidCache(fs::path(ImGuiFileDialog::Instance()->GetCurrentPath()));
        }
        ImGuiFileDialog::Instance()->Close();
    }
}

} // namespace lr
