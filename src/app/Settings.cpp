#include "Settings.hpp"

#include <imgui.h>
#include <nlohmann/json.hpp>

#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace lr
{
namespace
{
using json = nlohmann::json;

json encode(const SettingValue &value)
{
    return std::visit(
        [](const auto &item) -> json {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, glm::vec3>) return json::array({item.x, item.y, item.z});
            else return item;
        },
        value);
}

SettingValue decode(const json &value, const SettingValue &prototype, const std::string &where)
{
    try
    {
        return std::visit(
            [&](const auto &item) -> SettingValue {
                using T = std::decay_t<decltype(item)>;
                if constexpr (std::is_same_v<T, glm::vec3>)
                {
                    if (!value.is_array() || value.size() != 3) throw std::runtime_error("expected three numbers");
                    return glm::vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
                } else
                {
                    return value.get<T>();
                }
            },
            prototype);
    } catch (const std::exception &e)
    {
        throw std::runtime_error("Settings: invalid value at '" + where + "': " + e.what());
    }
}

bool drawSetting(Setting &setting)
{
    SettingValue value = setting.read();
    const bool enabled = !setting.enabled || setting.enabled();
    ImGui::BeginDisabled(!enabled);
    bool changed = std::visit(
        [&](auto &item) {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, bool>)
                return ImGui::Checkbox(setting.label.c_str(), &item);
            else if constexpr (std::is_same_v<T, int>)
                return ImGui::SliderInt(setting.label.c_str(), &item, static_cast<int>(setting.minimum),
                                        static_cast<int>(setting.maximum));
            else if constexpr (std::is_same_v<T, float>)
            {
                const char *format = setting.format.empty() ? "%.3f" : setting.format.c_str();
                if (setting.editor == SettingEditor::Drag)
                    return ImGui::DragFloat(setting.label.c_str(), &item, setting.speed, setting.minimum,
                                            setting.maximum, format);
                return ImGui::SliderFloat(setting.label.c_str(), &item, setting.minimum, setting.maximum, format,
                                          setting.logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
            } else if constexpr (std::is_same_v<T, glm::vec3>)
                return ImGui::ColorEdit3(setting.label.c_str(), &item.x);
        },
        value);
    ImGui::EndDisabled();
    if (changed) setting.write(value);
    return changed;
}
} // namespace

void Settings::add(SettingsCategory category)
{
    for (const auto &existing : m_categories)
        if (existing.key == category.key) throw std::invalid_argument("Settings: duplicate category key '" + category.key + "'");

    std::unordered_set<std::string> keys;
    for (const Setting &setting : category.settings)
    {
        if (!setting.read || !setting.write) throw std::invalid_argument("Settings: unbound setting '" + setting.key + "'");
        if (!keys.insert(setting.key).second) throw std::invalid_argument("Settings: duplicate setting key '" + setting.key + "'");
    }
    m_categories.push_back(std::move(category));
}

json Settings::serialize() const
{
    json categories = json::object();
    for (const SettingsCategory &category : m_categories)
    {
        json values = json::object();
        for (const Setting &setting : category.settings) values[setting.key] = encode(setting.read());
        categories[category.key] = std::move(values);
    }
    return {{"version", 1}, {"categories", std::move(categories)}};
}

void Settings::deserialize(const json &value)
{
    if (!value.is_object() || value.value("version", 0) != 1 || !value.contains("categories") ||
        !value["categories"].is_object())
        throw std::runtime_error("Settings: unsupported or malformed document");

    const json &categories = value["categories"];
    for (SettingsCategory &category : m_categories)
    {
        auto serializedCategory = categories.find(category.key);
        if (serializedCategory == categories.end()) continue;
        if (!serializedCategory->is_object())
            throw std::runtime_error("Settings: category '" + category.key + "' must be an object");

        std::vector<std::pair<Setting *, SettingValue>> decoded;
        for (Setting &setting : category.settings)
        {
            auto serializedSetting = serializedCategory->find(setting.key);
            if (serializedSetting != serializedCategory->end())
                decoded.emplace_back(&setting, decode(*serializedSetting, setting.read(), category.key + "." + setting.key));
        }
        for (auto &[setting, settingValue] : decoded) setting->write(settingValue);
        if (!decoded.empty() && category.onChanged) category.onChanged();
    }
}

bool drawSettings(Settings &settings)
{
    bool anyChanged = false;
    for (SettingsCategory &category : settings.categories())
    {
        if (!ImGui::CollapsingHeader(category.title.c_str(), category.defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen : 0))
            continue;
        ImGui::Indent();
        bool categoryChanged = false;
        if (category.drawExtra) category.drawExtra();
        for (Setting &setting : category.settings) categoryChanged |= drawSetting(setting);
        ImGui::Unindent();
        if (categoryChanged && category.onChanged) category.onChanged();
        anyChanged |= categoryChanged;
    }
    return anyChanged;
}

} // namespace lr
