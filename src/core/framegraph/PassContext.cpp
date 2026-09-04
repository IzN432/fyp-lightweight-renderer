#include "PassContext.hpp"

#include "FrameGraphDefinition.hpp"
#include "ResourceRegistry.hpp"

namespace lr
{

VkExtent2D PassContext::extent(ImageHandle image) const { return m_registry.getImageExtent(m_definition.name(image)); }

} // namespace lr
