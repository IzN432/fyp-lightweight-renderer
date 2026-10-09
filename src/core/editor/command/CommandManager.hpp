#pragma once

#include "Command.hpp"

#include <memory>
#include <vector>

namespace lr
{

class CommandManager
{
public:
    void executeCommand(std::unique_ptr<Command> command);
    void appendCommandWithoutExecuting(std::unique_ptr<Command> command);

    void undo();
    void redo();

    // One gesture that records several commands — a gizmo drag that also writes an Auto Key
    // keyframe, an inspector edit that does the same — collapses into one history entry, so one
    // undo takes the whole gesture back rather than half of it. Commands recorded while a
    // transaction is open are still executed as usual; only where they land in the history changes.
    //
    // Transactions nest, and an empty one leaves no trace. A gesture opens and closes its own
    // within the frame it completes in, which is what keeps undo() and the temporary histories
    // below from ever seeing a half-open one; use CommandTransaction rather than pairing these by
    // hand.
    void beginTransaction();
    void endTransaction();
    bool inTransaction() const { return !m_transactions.empty(); }

    void beginTemporaryHistory();
    void cancelTemporaryHistory();
    void replaceTemporaryHistory(std::unique_ptr<Command> command);
    bool hasTemporaryHistory() const { return m_histories.size() > 1; }

private:
    struct History
    {
        std::vector<std::unique_ptr<Command>> commands;
        std::vector<std::unique_ptr<Command>> redo;
    };

    History &activeHistory() { return m_histories.back(); }
    void rollbackActiveHistory();
    // Files a command under the innermost open transaction, or under the active history when there
    // is none.
    void record(std::unique_ptr<Command> command);

    std::vector<History> m_histories{1};
    // The commands each open transaction has collected so far, innermost last.
    std::vector<std::vector<std::unique_ptr<Command>>> m_transactions;
};

// Opens a transaction on `commands` for as long as it is in scope. Declare one at the top of the
// code that completes a gesture; everything recorded under it becomes a single undo.
class CommandTransaction
{
public:
    explicit CommandTransaction(CommandManager &commands) : m_commands(commands)
    {
        m_commands.beginTransaction();
    }

    ~CommandTransaction() { m_commands.endTransaction(); }

    CommandTransaction(const CommandTransaction &)            = delete;
    CommandTransaction &operator=(const CommandTransaction &) = delete;

private:
    CommandManager &m_commands;
};

} // namespace lr
