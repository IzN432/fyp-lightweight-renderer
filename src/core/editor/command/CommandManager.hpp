#pragma once

#include "Command.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace lr
{

class CommandManager
{
public:
    struct HistoryEntryId
    {
        uint64_t value;
        bool operator==(const HistoryEntryId &) const = default;
    };

    void executeCommand(std::unique_ptr<Command> command);
    void appendCommandWithoutExecuting(std::unique_ptr<Command> command);

    // Executes a deferred follow-up and folds it into its originating entry only while that entry
    // is still the undo head. Otherwise the command is recorded as a separate edit.
    bool executeCommandAndAppendTo(HistoryEntryId entry, std::unique_ptr<Command> command);

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
    HistoryEntryId beginTransaction();
    void endTransaction();
    bool inTransaction() const { return !m_transactions.empty(); }
    std::optional<HistoryEntryId> currentTransactionId() const;

    void beginTemporaryHistory();
    void cancelTemporaryHistory();
    void replaceTemporaryHistory(std::unique_ptr<Command> command);
    bool hasTemporaryHistory() const { return m_histories.size() > 1; }

private:
    struct HistoryEntry
    {
        HistoryEntryId          id;
        std::unique_ptr<Command> command;
    };

    struct History
    {
        std::vector<HistoryEntry> commands;
        std::vector<HistoryEntry> redo;
    };

    struct Transaction
    {
        HistoryEntryId                       id;
        std::vector<std::unique_ptr<Command>> commands;
    };

    History &activeHistory() { return m_histories.back(); }
    void rollbackActiveHistory();
    // Files a command under the innermost open transaction, or under the active history when there
    // is none.
    void record(std::unique_ptr<Command> command);

    std::vector<History> m_histories{1};
    // The commands each open transaction has collected so far, innermost last.
    std::vector<Transaction> m_transactions;
    uint64_t                 m_nextHistoryEntryId = 1;
};

// Opens a transaction on `commands` for as long as it is in scope. Declare one at the top of the
// code that completes a gesture; everything recorded under it becomes a single undo.
class CommandTransaction
{
public:
    explicit CommandTransaction(CommandManager &commands)
        : m_commands(commands), m_id(m_commands.beginTransaction())
    {
    }

    ~CommandTransaction() { m_commands.endTransaction(); }

    CommandTransaction(const CommandTransaction &)            = delete;
    CommandTransaction &operator=(const CommandTransaction &) = delete;

    CommandManager::HistoryEntryId id() const { return m_id; }

private:
    CommandManager                &m_commands;
    CommandManager::HistoryEntryId m_id;
};

} // namespace lr
