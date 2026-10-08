#pragma once

#include <glm/vec3.hpp>
#include <nlohmann/json_fwd.hpp>

#include <functional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lr
{

using SettingValue = std::variant<bool, int, float, glm::vec3>;

enum class SettingEditor
{
    Checkbox,
    Slider,
    Drag,
    Color3,
};

struct Setting
{
    std::string                   key;
    std::string                   label;
    SettingEditor                 editor = SettingEditor::Slider;
    std::function<SettingValue()> read;
    std::function<void(const SettingValue &)> write;
    std::function<bool()>         enabled;
    float                         minimum = 0.0f;
    float                         maximum = 0.0f;
    float                         speed   = 1.0f;
    std::string                   format;
    bool                          logarithmic = false;

    template <typename T>
    static Setting bind(std::string key, std::string label, SettingEditor editor, T &value)
    {
        return Setting{
            .key    = std::move(key),
            .label  = std::move(label),
            .editor = editor,
            .read   = [&value] { return SettingValue(value); },
            .write  = [&value](const SettingValue &newValue) { value = std::get<T>(newValue); },
            .enabled = [] { return true; },
        };
    }
};

struct SettingsCategory
{
    std::string           key;
    std::string           title;
    bool                  defaultOpen = false;
    std::vector<Setting>  settings;
    std::function<void()> drawExtra;
    std::function<void()> onChanged;
};

// One data-driven representation is shared by the settings UI and persistence. Keys are stable
// serialization identifiers; titles and labels are presentation-only and may be changed freely.
class Settings
{
public:
    void add(SettingsCategory category);

    std::vector<SettingsCategory>       &categories() { return m_categories; }
    const std::vector<SettingsCategory> &categories() const { return m_categories; }

    nlohmann::json serialize() const;
    void           deserialize(const nlohmann::json &value);

private:
    std::vector<SettingsCategory> m_categories;
};

// Returns true when at least one setting changed. Per-category change callbacks run once after all
// widgets in that category have been drawn.
bool drawSettings(Settings &settings);

} // namespace lr
