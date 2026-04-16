#pragma once

#include <vulkan/vulkan.h>

namespace lr
{

class Pipeline
{
public:
    virtual ~Pipeline() = default;

    virtual VkPipeline          get()       const = 0;
    virtual VkPipelineBindPoint bindPoint() const = 0;

    Pipeline(const Pipeline &) = delete;
    Pipeline &operator=(const Pipeline &) = delete;

protected:
    Pipeline() = default;
};

}  // namespace lr
