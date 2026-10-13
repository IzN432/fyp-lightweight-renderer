#include "core/Paths.hpp"

#include <system_error>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <vector>
#else
#include <unistd.h>
#endif

namespace lr::paths
{
namespace
{

// The directory holding the running executable. Empty if the platform query fails, which only
// pushes the callers below onto their build-time fallback.
std::filesystem::path executableDirectory()
{
#if defined(_WIN32)
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;)
    {
        const DWORD written = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (written == 0)
        {
            return {};
        }
        // The call truncates rather than reporting the length it needs, so a full buffer means
        // "try again with more room".
        if (written < buffer.size())
        {
            buffer.resize(written);
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
    return std::filesystem::path(buffer).parent_path();
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0)
    {
        return {};
    }
    return std::filesystem::path(buffer.data()).parent_path();
#else
    std::error_code    ec;
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec)
    {
        return {};
    }
    return self.parent_path();
#endif
}

// `relative` next to the executable when that exists, otherwise `buildTimeFallback`. The
// existence check is what lets one binary serve both layouts: an installed build ships its
// shaders and assets alongside itself, while a build-tree run finds nothing there and uses the
// directories CMake pointed at.
std::filesystem::path resolve(const char *relative, const char *buildTimeFallback)
{
    const std::filesystem::path exeDir = executableDirectory();
    if (!exeDir.empty())
    {
        std::error_code    ec;
        const std::filesystem::path candidate = exeDir / relative;
        if (std::filesystem::is_directory(candidate, ec))
        {
            return candidate;
        }
    }
    return std::filesystem::path(buildTimeFallback);
}

} // namespace

const std::filesystem::path &shaderDirectory()
{
    static const std::filesystem::path dir = resolve("shaders", LR_SHADER_DIR);
    return dir;
}

const std::filesystem::path &assetDirectory()
{
    static const std::filesystem::path dir = resolve("assets", LR_ASSET_DIR);
    return dir;
}

} // namespace lr::paths
