#include "PassContext.hpp"

#include "FrameGraphDefinition.hpp"
#include "PassDefinition.hpp"
#include "ResourceRegistry.hpp"

namespace lr
{

VkExtent2D PassContext::extent(ImageHandle image) const { return m_registry.getImageExtent(m_definition.name(image)); }

uint32_t PassContext::pushConstantSize() const { return m_pass.pushConstantSize; }

VkShaderStageFlags PassContext::pushConstantStages() const { return m_pass.pushConstantStages; }

} // namespace lr
