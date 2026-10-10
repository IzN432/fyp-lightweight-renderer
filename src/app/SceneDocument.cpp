#include "SceneDocument.hpp"
#include "EditorSession.hpp"

#include "core/app/Viewer.hpp"
#include "core/scene/SceneManager.hpp"
#include "features/rigid_body/PhysicsWorld.hpp"

#include <ImGuiFileDialog.h>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include <exception>
#include <utility>

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

constexpr const char *kSceneExtension = ".lrscene";

constexpr ImVec2 kDialogSize(640.0f, 360.0f);

ImGuiFileDialogFlags commonDialogFlags()
{
    return ImGuiFileDialogFlags_Modal | ImGuiFileDialogFlags_CaseInsensitiveExtentionFiltering |
           ImGuiFileDialogFlags_ShowDevicesButton;
}
} // namespace

SceneDocument::SceneDocument(Viewer &viewer, SceneManager &sceneManager, EditorSession &editor,
                             PhysicsWorld &physicsWorld, SceneLoaderConfig importConfig)
    : m_viewer(viewer), m_sceneManager(sceneManager), m_editor(editor), m_physicsWorld(physicsWorld),
      m_importConfig(std::move(importConfig))
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
                spdlog::error("Failed to save scene: {}", e.what());
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
                spdlog::error("Failed to import scene: {}", e.what());
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }
}

} // namespace lr
