#pragma once

#include "core/scene/SceneManager.hpp"

namespace lr
{

// Owns the interactive Laplace-Beltrami analysis workflow: calculation, analysis-mode selection,
// GUI, and stale-result tracking after geometry edits.
class LaplaceBeltramiTool
{
public:
    LaplaceBeltramiTool(const Mesh &mesh, SceneManager &sceneManager);

    void rebind(const Mesh &mesh);

    void onGui();

private:
    void calculate();
    const Mesh *m_mesh = nullptr;
    Mesh::Revision m_resultPositionsRevision = 0;
    Mesh::Revision m_resultTopologyRevision = 0;
    SceneManager &m_sceneManager;

    bool m_hasResult = false;
};

} // namespace lr
