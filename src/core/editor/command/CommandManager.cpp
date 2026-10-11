#include "CommandManager.hpp"

#include "CompositeCommand.hpp"

#include <utility>

namespace lr
{

void CommandManager::record(std::unique_ptr<Command> command)
{
    if (!m_transactions.empty())
    {
        m_transactions.back().commands.push_back(std::move(command));
        return;
    }
    activeHistory().commands.push_back({HistoryEntryId{m_nextHistoryEntryId++}, std::move(command)});
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

bool CommandManager::executeCommandAndAppendTo(HistoryEntryId entry,
                                               std::unique_ptr<Command> command)
{
    command->execute();
    History &history = activeHistory();
    if (inTransaction() || history.commands.empty() || history.commands.back().id != entry)
    {
        record(std::move(command));
        return false;
    }

    std::vector<std::unique_ptr<Command>> combined;
    combined.push_back(std::move(history.commands.back().command));
    combined.push_back(std::move(command));
    history.commands.back().command = std::make_unique<CompositeCommand>(std::move(combined));
    history.redo.clear();
    return true;
}

CommandManager::HistoryEntryId CommandManager::beginTransaction()
{
    const HistoryEntryId id = m_transactions.empty()
                                  ? HistoryEntryId{m_nextHistoryEntryId++}
                                  : m_transactions.front().id;
    m_transactions.push_back({id, {}});
    return id;
}

std::optional<CommandManager::HistoryEntryId> CommandManager::currentTransactionId() const
{
    if (m_transactions.empty()) return std::nullopt;
    return m_transactions.front().id;
}

void CommandManager::endTransaction()
{
    if (m_transactions.empty())
    {
        return;
    }

    Transaction transaction = std::move(m_transactions.back());
    m_transactions.pop_back();
    if (transaction.commands.empty())
    {
        return;
    }

    // A gesture that recorded one command is that command: wrapping it would only add a layer for
    // undo to walk through.
    std::unique_ptr<Command> command = transaction.commands.size() == 1
                                           ? std::move(transaction.commands.front())
                                           : std::make_unique<CompositeCommand>(
                                                 std::move(transaction.commands));
    if (!m_transactions.empty())
    {
        m_transactions.back().commands.push_back(std::move(command));
        return;
    }
    activeHistory().commands.push_back({transaction.id, std::move(command)});
    activeHistory().redo.clear();
}

void CommandManager::undo()
{
    History &history = activeHistory();
    if (!history.commands.empty())
    {
        HistoryEntry entry = std::move(history.commands.back());
        entry.command->undo();
        history.commands.pop_back();
        history.redo.push_back(std::move(entry));
    }
}

void CommandManager::redo()
{
    History &history = activeHistory();
    if (!history.redo.empty())
    {
        HistoryEntry entry = std::move(history.redo.back());
        history.redo.pop_back();
        entry.command->execute();
        history.commands.push_back(std::move(entry));
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
        history.commands.back().command->undo();
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
