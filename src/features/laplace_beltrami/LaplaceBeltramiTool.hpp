#pragma once

#include "core/editor/EditorTool.hpp"
#include "core/scene/SceneManager.hpp"

#include <string_view>

namespace lr
{

// Owns the interactive Laplace-Beltrami analysis workflow: calculation, the heatmap editor state,
// GUI, and stale-result tracking after geometry edits.
class LaplaceBeltramiTool final : public EditorTool
{
public:
    explicit LaplaceBeltramiTool(SceneManager &sceneManager);

    // The editor state this tool registers, for hosts that want to bind a shortcut to it.
    static std::string_view stateId();

    void registerWith(EditorServices &services) override;

    // Analysis magnitudes are per-mesh, so a new target discards the previous result.
    void onTargetChanged(const EditableMeshContext &target) override;
    void onTargetCleared() override;

    void        drawPanel() override;
    const char *displayName() const override { return "Laplace-Beltrami"; }

private:
    void calculate();
    void discardResult();

    const Mesh            *m_mesh   = nullptr;
    EditorStateController *m_states = nullptr;

    Mesh::Revision m_resultPositionsRevision = 0;
    Mesh::Revision m_resultTopologyRevision  = 0;
    SceneManager  &m_sceneManager;

    bool m_hasResult = false;
};

} // namespace lr
