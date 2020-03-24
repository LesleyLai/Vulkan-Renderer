#include "logical_device.hpp"
#include "instance.hpp"
#include "panic.hpp"
#include "physical_device_picker.hpp"
#include "utils.hpp"

namespace vkh {

[[nodiscard]] auto
create_logical_device(VkPhysicalDevice pd,
                      const vkh::QueueFamilyIndices& indices) noexcept
    -> VkDevice
{
  const auto unique_indices = indices.to_set();

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  queue_create_infos.resize(unique_indices.size());

  float queue_priority = 1.0f;
  std::transform(std::begin(unique_indices), std::end(unique_indices),
                 std::begin(queue_create_infos), [&](uint32_t index) {
                   return VkDeviceQueueCreateInfo{
                       .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                       .pNext = nullptr,
                       .flags = 0,
                       .queueFamilyIndex = index,
                       .queueCount = 1,
                       .pQueuePriorities = &queue_priority,
                   };
                 });

  const VkPhysicalDeviceFeatures features = {};

  const VkDeviceCreateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queueCreateInfoCount = vkh::to_u32(queue_create_infos.size()),
      .pQueueCreateInfos = queue_create_infos.data(),
#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
      .enabledLayerCount = vkh::to_u32(vkh::validation_layers.size()),
      .ppEnabledLayerNames = vkh::validation_layers.data(),
#else
      .enabledLayerCount = 0,
      .ppEnabledLayerNames = nullptr,
#endif
      .enabledExtensionCount = vkh::to_u32(device_extensions.size()),
      .ppEnabledExtensionNames = device_extensions.data(),
      .pEnabledFeatures = &features,
  };

  VkDevice device = nullptr;
  if (vkCreateDevice(pd, &create_info, nullptr, &device) != VK_SUCCESS) {
    vkh::panic("Vulkan: failed to create logical device!");
  }

  return device;
}

} // namespace vkh