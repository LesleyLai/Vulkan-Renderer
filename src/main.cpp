#include <fmt/format.h>

#include <volk.h>

#include "beyond/core/utils/panic.hpp"

#include "vulkan_helper/instance.hpp"

#include "window.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

namespace {

// constexpr std::array device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

//[[nodiscard]] auto
// check_device_extension_support(VkPhysicalDevice device) noexcept -> bool
//{
//  const auto available = get_vector_with<VkExtensionProperties>(
//      [device](uint32_t* count, VkExtensionProperties* data) {
//        vkEnumerateDeviceExtensionProperties(device, nullptr, count, data);
//      });
//
//  std::set<std::string> required(device_extensions.begin(),
//                                 device_extensions.end());
//
//  for (const auto& extension : available) {
//    required.erase(static_cast<const char*>(extension.extensionName));
//  }
//
//  return required.empty();
//}
//
//// Higher is better, negative means not suitable
//[[nodiscard]] auto rate_physical_device(VkPhysicalDevice device,
//                                        VkSurfaceKHR surface) noexcept -> int
//{
//  using namespace beyond::graphics::vulkan;
//
//  static constexpr int failing_score = -1000;
//
//  // If cannot find indices for all the queues, return -1000
//  const auto maybe_indices = find_queue_families(device, surface);
//  if (!maybe_indices) {
//    return failing_score;
//  }
//
//  // If not support extension, return -1000
//  if (!check_device_extension_support(device)) {
//    return failing_score;
//  }
//
//  // If swapchain not adequate, return -1000
//  const auto swapchain_support = query_swapchain_support(device, surface);
//  if (swapchain_support.formats.empty() ||
//      swapchain_support.present_modes.empty()) {
//    return failing_score;
//  }
//
//  VkPhysicalDeviceProperties properties;
//  vkGetPhysicalDeviceProperties(device, &properties);
//
//  VkPhysicalDeviceFeatures features;
//  vkGetPhysicalDeviceFeatures(device, &features);
//
//  // Biased toward discrete GPU
//  int score = 0;
//  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
//    score += 100;
//  }
//
//  return score;
//}
} // namespace

auto main() -> int
{
  if (volkInitialize() != VK_SUCCESS) {
    beyond::panic("Cannot find a Vulkan Loader in the system!");
  }

  Window window(1024, 768, "Vulkan Renderer");

  [[maybe_unused]] auto instance = vkh::create_instance(
      window.title().c_str(), window.get_required_instance_extensions());

  while (!window.should_close()) {
    window.poll_events();
    window.swap_buffers();
  }
}
