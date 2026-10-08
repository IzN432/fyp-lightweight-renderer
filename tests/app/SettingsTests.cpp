#include "app/Settings.hpp"

#include <nlohmann/json.hpp>

#include <cassert>
#include <stdexcept>

namespace
{
void roundTripsEverySupportedValue()
{
    bool        enabled  = true;
    int         samples  = 8;
    float       exposure = 1.25f;
    glm::vec3   color(0.1f, 0.2f, 0.3f);
    int         changes = 0;

    lr::Settings settings;
    settings.add({
        .key = "rendering",
        .title = "Rendering",
        .settings = {
            lr::Setting::bind("enabled", "Enabled", lr::SettingEditor::Checkbox, enabled),
            lr::Setting::bind("samples", "Samples", lr::SettingEditor::Slider, samples),
            lr::Setting::bind("exposure", "Exposure", lr::SettingEditor::Slider, exposure),
            lr::Setting::bind("color", "Color", lr::SettingEditor::Color3, color),
        },
        .onChanged = [&] { ++changes; },
    });

    const nlohmann::json saved = settings.serialize();
    enabled = false;
    samples = 1;
    exposure = 0.0f;
    color = glm::vec3(0.0f);
    settings.deserialize(saved);

    assert(enabled && samples == 8 && exposure == 1.25f);
    assert(color == glm::vec3(0.1f, 0.2f, 0.3f));
    assert(changes == 1);
}

void ignoresUnknownAndRejectsWrongTypes()
{
    float value = 2.0f;
    lr::Settings settings;
    settings.add({.key = "known", .title = "Known",
                  .settings = {lr::Setting::bind("value", "Value", lr::SettingEditor::Slider, value)}});

    settings.deserialize({{"version", 1}, {"categories", {{"unknown", {{"anything", 3}}}}}});
    assert(value == 2.0f);

    bool threw = false;
    try
    {
        settings.deserialize({{"version", 1}, {"categories", {{"known", {{"value", "wrong"}}}}}});
    } catch (const std::runtime_error &)
    {
        threw = true;
    }
    assert(threw && value == 2.0f);
}
} // namespace

int main()
{
    roundTripsEverySupportedValue();
    ignoresUnknownAndRejectsWrongTypes();
    return 0;
}
