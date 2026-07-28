#pragma once

#include "Command.hpp"
#include "core/editor/VertexManager.hpp"

#include <vector>
#include <glm/vec3.hpp>

namespace lr
{

class TranslatePointsCommand : public Command
{
public:
    TranslatePointsCommand(VertexManager &vertexManager, std::vector<uint32_t> indices, glm::vec3 translation)
        : m_vertexManager(vertexManager), m_indices(std::move(indices)), m_translation(translation) {}

    void execute() override;
    void undo() override;
private:
    VertexManager &m_vertexManager;
    std::vector<uint32_t> m_indices;
    glm::vec3 m_translation;
};

}