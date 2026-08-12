#include "ArapDeformCommand.hpp"

namespace lr
{

void ArapDeformCommand::execute()
{
    m_vertexManager.setPositions(m_after);
}

void ArapDeformCommand::undo()
{
    m_vertexManager.setPositions(m_before);
}

}  // namespace lr
