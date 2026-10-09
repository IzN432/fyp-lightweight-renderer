#include "CommandManager.hpp"

#include "CompositeCommand.hpp"

#include <utility>

namespace lr
{

void CommandManager::record(std::unique_ptr<Command> command)
{
    if (!m_transactions.empty())
    {
        m_transactions.back().push_back(std::move(command));
        return;
    }
    activeHistory().commands.push_back(std::move(command));
    activeHistory().redo.clear();
}

void CommandManager::executeCommand(std::unique_ptr<Command> command)
{
    command->execute();
    record(std::move(command));
}
void CommandManager::appendCommandWithoutExecuting(std::unique_ptr<Command> command)
{
    record(std::move(command));
}

void CommandManager::beginTransaction() { m_transactions.emplace_back(); }

void CommandManager::endTransaction()
{
    if (m_transactions.empty())
    {
        return;
    }

    std::vector<std::unique_ptr<Command>> collected = std::move(m_transactions.back());
    m_transactions.pop_back();
    if (collected.empty())
    {
        return;
    }

    // A gesture that recorded one command is that command: wrapping it would only add a layer for
    // undo to walk through.
    std::unique_ptr<Command> entry = collected.size() == 1
                                         ? std::move(collected.front())
                                         : std::make_unique<CompositeCommand>(std::move(collected));
    record(std::move(entry));
}

void CommandManager::undo()
{
    History &history = activeHistory();
    if (!history.commands.empty())
    {
        std::unique_ptr<Command> command = std::move(history.commands.back());
        command->undo();
        history.commands.pop_back();
        history.redo.push_back(std::move(command));
    }
}

void CommandManager::redo()
{
    History &history = activeHistory();
    if (!history.redo.empty())
    {
        std::unique_ptr<Command> command = std::move(history.redo.back());
        history.redo.pop_back();
        command->execute();
        history.commands.push_back(std::move(command));
    }
}

void CommandManager::beginTemporaryHistory()
{
    m_histories.emplace_back();
}

void CommandManager::rollbackActiveHistory()
{
    History &history = activeHistory();
    while (!history.commands.empty())
    {
        history.commands.back()->undo();
        history.commands.pop_back();
    }
}

void CommandManager::cancelTemporaryHistory()
{
    if (!hasTemporaryHistory())
    {
        return;
    }
    rollbackActiveHistory();
    m_histories.pop_back();
}

void CommandManager::replaceTemporaryHistory(std::unique_ptr<Command> command)
{
    if (!hasTemporaryHistory())
    {
        executeCommand(std::move(command));
        return;
    }
    rollbackActiveHistory();
    m_histories.pop_back();
    executeCommand(std::move(command));
}

} // namespace lr
