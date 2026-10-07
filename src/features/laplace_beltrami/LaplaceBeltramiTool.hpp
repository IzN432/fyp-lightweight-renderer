#pragma once

#include "core/scene/SceneManager.hpp"

#include <functional>

namespace lr
{

// Owns the interactive Laplace-Beltrami analysis workflow: calculation, analysis-mode selection,
// GUI, and stale-result tracking after geometry edits.
class LaplaceBeltramiTool
{
public:
    LaplaceBeltramiTool(const Mesh &mesh, SceneManager &sceneManager);

    void rebind(const Mesh &mesh);

    // Draws the tool's GUI. setEnabled is called when the user toggles the analysis mode on/off.
    void onGui(bool enabled, const std::function<void(bool)> &setEnabled);

private:
    void calculate();
    const Mesh *m_mesh = nullptr;
    Mesh::Revision m_resultPositionsRevision = 0;
    Mesh::Revision m_resultTopologyRevision = 0;
    SceneManager &m_sceneManager;

    bool m_hasResult = false;
};

} // namespace lr
