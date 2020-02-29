#include <fmt/format.h>

#include <volk.h>

#include <set>

#include "beyond/core/utils/panic.hpp"

#include "vulkan_helper/instance.hpp"
#include "vulkan_helper/queue_indices.hpp"
#include "vulkan_helper/swapchain.hpp"
#include "vulkan_helper/utils.hpp"

#include "window.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

namespace {

constexpr std::array device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

[[nodiscard]] auto
check_device_extension_support(VkPhysicalDevice device) noexcept -> bool
{
  const auto available = vkh::get_vector_with<VkExtensionProperties>(
      [device](uint32_t* count, VkExtensionProperties* data) {
        vkEnumerateDeviceExtensionProperties(device, nullptr, count, data);
      });

  std::set<std::string> required(device_extensions.begin(),
                                 device_extensions.end());

  for (const auto& extension : available) {
    required.erase(static_cast<const char*>(extension.extensionName));
  }

  return required.empty();
}

// Higher is better, negative means not suitable
[[nodiscard]] auto rate_physical_device(VkPhysicalDevice device,
                                        VkSurfaceKHR surface) noexcept -> int
{
  static constexpr int failing_score = -1000;

  // If cannot find indices for all the queues, return -1000
  const auto maybe_indices = vkh::find_queue_families(device, surface);
  if (!maybe_indices) {
    return failing_score;
  }

  // If not support extension, return -1000
  if (!check_device_extension_support(device)) {
    return failing_score;
  }

  // If swapchain not adequate, return -1000
  const auto swapchain_support = vkh::query_swapchain_support(device, surface);
  if (swapchain_support.formats.empty() ||
      swapchain_support.present_modes.empty()) {
    return failing_score;
  }

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(device, &properties);

  VkPhysicalDeviceFeatures features;
  vkGetPhysicalDeviceFeatures(device, &features);

  // Biased toward discrete GPU
  int score = 0;
  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score += 100;
  }

  return score;
}

[[nodiscard]] auto pick_physical_device(VkInstance instance,
                                        VkSurfaceKHR surface) noexcept
    -> VkPhysicalDevice
{
  const auto available_devices = vkh::get_vector_with<VkPhysicalDevice>(
      [instance](uint32_t* count, VkPhysicalDevice* data) {
        return vkEnumeratePhysicalDevices(instance, count, data);
      });
  if (available_devices.empty()) {
    beyond::panic("failed to find GPUs with Vulkan support!");
  }

  using ScoredPair = std::pair<int, VkPhysicalDevice>;
  std::vector<ScoredPair> scored_pairs;
  scored_pairs.reserve(available_devices.size());
  for (const auto& device : available_devices) {
    const auto score = rate_physical_device(device, surface);
    if (score > 0) {
      scored_pairs.emplace_back(score, device);
    }
  }

  if (scored_pairs.empty()) {
    beyond::panic(
        "Vulkan failed to find GPUs with enough nessesory graphics support!");
  }

  std::sort(std::begin(scored_pairs), std::end(scored_pairs),
            [](const ScoredPair& lhs, const ScoredPair& rhs) {
              return lhs.first > rhs.first;
            });

  const auto physical_device = scored_pairs.front().second;

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(physical_device, &properties);
  fmt::print("GPU: {}\n", properties.deviceName);
  std::fflush(stdout);

  // Returns the pair with highest score
  return physical_device;
}

} // anonymous namespace

auto main() -> int
{
  if (volkInitialize() != VK_SUCCESS) {
    beyond::panic("Cannot find a Vulkan Loader in the system!");
  }

  Window window(1024, 768, "Vulkan Renderer");

  auto instance = vkh::create_instance(
      window.title().c_str(), window.get_required_instance_extensions());

  VkSurfaceKHR surface;
  window.create_vulkan_surface(instance, nullptr, surface);

#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
  auto debug_messenger = vkh::create_debug_messenger(instance);
#endif

  const auto physical_device = pick_physical_device(instance, surface);
  [[maybe_unused]] const auto queue_family_indices =
      vkh::find_queue_families(physical_device, surface);

  while (!window.should_close()) {
    window.poll_events();
    window.swap_buffers();
  }

#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
  vkDestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
#endif
  vkDestroySurfaceKHR(instance, surface, nullptr);
  vkDestroyInstance(instance, nullptr);
}
