#include "core/editor/command/Command.hpp"
#include "core/editor/command/CommandManager.hpp"

#include <cassert>
#include <memory>
#include <string>

namespace
{

// Appends to a shared log on execute and undo, so the tests can check both that a gesture is one
// history entry and that its parts are undone in the reverse of the order they were recorded.
class LoggingCommand final : public lr::Command
{
public:
    LoggingCommand(std::string &log, std::string name) : m_log(log), m_name(std::move(name)) {}

    void execute() override { m_log += "+" + m_name; }
    void undo() override { m_log += "-" + m_name; }

private:
    std::string &m_log;
    std::string  m_name;
};

std::unique_ptr<lr::Command> logging(std::string &log, std::string name)
{
    return std::make_unique<LoggingCommand>(log, std::move(name));
}

} // namespace

int main()
{
    // A gesture that records two commands is one undo, and its parts come back in reverse order.
    {
        std::string        log;
        lr::CommandManager commands;
        {
            lr::CommandTransaction transaction(commands);
            assert(commands.inTransaction());
            commands.executeCommand(logging(log, "transform"));
            commands.executeCommand(logging(log, "keyframe"));
        }
        assert(!commands.inTransaction());
        assert(log == "+transform+keyframe");

        log.clear();
        commands.undo();
        assert(log == "-keyframe-transform");

        // And one redo puts the whole gesture back.
        log.clear();
        commands.redo();
        assert(log == "+transform+keyframe");

        // Only the one entry: a second undo has nothing left to take back.
        log.clear();
        commands.undo();
        commands.undo();
        assert(log == "-keyframe-transform");
    }

    // A command appended rather than executed inside a transaction is not run on the way in, which
    // is how the inspector and the gizmo handlers record an edit their widgets already applied.
    {
        std::string        log;
        lr::CommandManager commands;
        {
            lr::CommandTransaction transaction(commands);
            commands.appendCommandWithoutExecuting(logging(log, "edit"));
        }
        assert(log.empty());
        commands.undo();
        assert(log == "-edit");
    }

    // An empty transaction leaves no trace, which is what most frames of a component's inspector
    // produce.
    {
        std::string        log;
        lr::CommandManager commands;
        commands.executeCommand(logging(log, "first"));
        {
            lr::CommandTransaction transaction(commands);
        }
        log.clear();
        commands.undo();
        assert(log == "-first");
    }

    // A transaction that recorded one command is that command, not a wrapper around it.
    {
        std::string        log;
        lr::CommandManager commands;
        {
            lr::CommandTransaction transaction(commands);
            commands.executeCommand(logging(log, "only"));
        }
        log.clear();
        commands.undo();
        assert(log == "-only");
        commands.redo();
    }

    // Nesting: the inner gesture becomes part of the outer one rather than its own entry.
    {
        std::string        log;
        lr::CommandManager commands;
        {
            lr::CommandTransaction outer(commands);
            commands.executeCommand(logging(log, "outer"));
            {
                lr::CommandTransaction inner(commands);
                commands.executeCommand(logging(log, "innerA"));
                commands.executeCommand(logging(log, "innerB"));
            }
        }
        log.clear();
        commands.undo();
        assert(log == "-innerB-innerA-outer");
        log.clear();
        commands.undo();
        assert(log.empty());
    }

    // Recording a command after a transaction has closed starts a new entry, so an edit made later
    // is undone on its own.
    {
        std::string        log;
        lr::CommandManager commands;
        {
            lr::CommandTransaction transaction(commands);
            commands.executeCommand(logging(log, "gesture"));
        }
        commands.executeCommand(logging(log, "later"));

        log.clear();
        commands.undo();
        assert(log == "-later");
        log.clear();
        commands.undo();
        assert(log == "-gesture");
    }

    return 0;
}
