#pragma once
#include <filesystem>

namespace lr
{
namespace paths
{
    inline const std::filesystem::path shaderDir = LR_SHADER_DIR;
    inline const std::filesystem::path assetDir = LR_ASSET_DIR;

    inline const std::filesystem::path brdfLutPath = assetDir / "brdf_lut.png";
}
}