#include "CommandManager.hpp"

namespace lr
{

void CommandManager::executeCommand(std::unique_ptr<Command> command)
{
    command->execute();
    m_commandHistory.push_back(std::move(command));
    m_redoStack.clear();
}
void CommandManager::appendCommandWithoutExecuting(std::unique_ptr<Command> command)
{
    m_commandHistory.push_back(std::move(command));
    m_redoStack.clear();
}

void CommandManager::undo()
{
    if (!m_commandHistory.empty())
    {
        std::unique_ptr<Command> command = std::move(m_commandHistory.back());
        command->undo();
        m_commandHistory.pop_back();
        m_redoStack.push_back(std::move(command));
    }
}

void CommandManager::redo()
{
    if (!m_redoStack.empty())
    {
        std::unique_ptr<Command> command = std::move(m_redoStack.back());
        m_redoStack.pop_back();
        command->execute();
        m_commandHistory.push_back(std::move(command));
    }
}

}