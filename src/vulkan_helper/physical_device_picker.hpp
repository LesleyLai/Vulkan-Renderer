#ifndef VULKAN_RENDERER_PHYSICAL_DEVICE_PICKER_HPP
#define VULKAN_RENDERER_PHYSICAL_DEVICE_PICKER_HPP

#include <volk.h>

#include <array>

namespace vkh {

[[maybe_unused]] constexpr std::array device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

[[nodiscard]] auto pick_physical_device(VkInstance instance,
                                        VkSurfaceKHR surface) noexcept
    -> VkPhysicalDevice;

} // namespace vkh

#endif // VULKAN_RENDERER_PHYSICAL_DEVICE_PICKER_HPP
