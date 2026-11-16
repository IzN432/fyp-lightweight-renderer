#pragma once

#include "ArapSurfaceFrames.hpp"

#include "core/editor/command/Command.hpp"
#include "core/editor/VertexManager.hpp"

#include <cstdint>
#include <vector>
#include <glm/vec3.hpp>

namespace lr
{

// Undo for an ARAP drag. Unlike TranslatePointsCommand (one shared delta over a fixed index set),
// an ARAP solve can move a different, data-dependent subset of vertices by varying amounts each
// time, so this stores an explicit sparse before/after diff instead. The normals and tangents the
// drag rewrote are stored whole, so undo restores the exact (possibly authored) originals rather
// than recomputing them.
class ArapDeformCommand : public Command
{
public:
    struct VertexDiff
    {
        uint32_t  index;
        glm::vec3 before;
        glm::vec3 after;
    };

    ArapDeformCommand(VertexManager &vertexManager, std::vector<VertexDiff> diffs, SurfaceFrames framesBefore,
                      SurfaceFrames framesAfter)
        : m_vertexManager(vertexManager), m_diffs(std::move(diffs)), m_framesBefore(std::move(framesBefore)),
          m_framesAfter(std::move(framesAfter))
    {}

    void execute() override;
    void undo() override;

private:
    VertexManager          &m_vertexManager;
    std::vector<VertexDiff> m_diffs;
    SurfaceFrames           m_framesBefore;
    SurfaceFrames           m_framesAfter;
};

} // namespace lr
