#ifndef VULKAN_RENDERER_LOGICAL_DEVICE_HPP
#define VULKAN_RENDERER_LOGICAL_DEVICE_HPP

#include <volk.h>

#include "queue_indices.hpp"

namespace vkh {

[[nodiscard]] auto
create_logical_device(VkPhysicalDevice pd,
                      const vkh::QueueFamilyIndices& indices) noexcept
    -> VkDevice;

}

#endif // VULKAN_RENDERER_LOGICAL_DEVICE_HPP
