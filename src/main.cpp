#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <vector>

#include <volk.h>

#include "beyond/core/utils/panic.hpp"

#include "vulkan_queue_indices.hpp"
#include "vulkan_swapchain.hpp"
#include "vulkan_utils.hpp"

#include "window.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

namespace {

template <typename T, typename F> auto get_vector_with(F func) -> std::vector<T>
{
  std::uint32_t count;
  func(&count, nullptr);

  std::vector<T> vec(count);
  func(&count, vec.data());

  return vec;
}

template <typename T> constexpr auto to_u32(T value) noexcept -> std::uint32_t
{
  return static_cast<std::uint32_t>(value);
}

constexpr std::array validation_layers = {"VK_LAYER_KHRONOS_validation"};
constexpr std::array device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef VULKAN_RENDERER_ENABLE_VALIDATION_LAYER
constexpr bool enable_validation_layers = true;
#else
constexpr bool enable_validation_layers = false;
#endif

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

#ifdef VULKAN_RENDERER_ENABLE_VALIDATION_LAYER
VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
               VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
               const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
               void* /*pUserData*/)
{
  fmt::print("validation layer: {}\n", p_callback_data->pMessage);
  return VK_FALSE;
}

constexpr auto populate_debug_messenger_create_info() noexcept
    -> VkDebugUtilsMessengerCreateInfoEXT
{
  return {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .pNext = nullptr,
      .flags = 0,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = debug_callback,
      .pUserData = nullptr,
  };
}
#endif
} // namespace

auto check_validation_layer_support() noexcept -> bool
{
  const auto available =
      get_vector_with<VkLayerProperties>(vkEnumerateInstanceLayerProperties);

  return std::all_of(
      std::begin(validation_layers), std::end(validation_layers),
      [&](const char* layer_name) {
        return std::find_if(std::begin(available), std::end(available),
                            [&](const auto& layer_properties) {
                              return strcmp(layer_name,
                                            static_cast<const char*>(
                                                layer_properties.layerName));
                            }) != std::end(available);
      });
}

[[nodiscard]] auto create_instance(const Window& window) noexcept -> VkInstance
{
  if (enable_validation_layers && !check_validation_layer_support()) {
    beyond::panic("validation layers requested, but not available!");
  }

  VkInstance instance;

  const VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = nullptr,
      .pApplicationName = window.title().c_str(),
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "Beyond Game Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_1,
  };

  auto extensions = window.get_required_instance_extensions();
#ifdef VULKAN_RENDERER_ENABLE_VALIDATION_LAYER
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = beyond::vkh::to_u32(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

#ifdef VULKAN_RENDERER_ENABLE_VALIDATION_LAYER
  create_info.enabledLayerCount =
      static_cast<uint32_t>(validation_layers.size());
  create_info.ppEnabledLayerNames = validation_layers.data();

  VkDebugUtilsMessengerCreateInfoEXT debug_create_info =
      populate_debug_messenger_create_info();
  create_info.pNext = &debug_create_info;
#else
  create_info.enabledLayerCount = 0;
  create_info.pNext = nullptr;
#endif

  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    beyond::panic("Cannot create vulkan instance!");
  }

  return instance;
} // anonymous namespace

auto main() -> int
{
  if (volkInitialize() != VK_SUCCESS) {
    beyond::panic("Cannot find a Vulkan Loader in the system!");
  }

  Window window(1024, 768, "Vulkan Renderer");
  auto instance = create_instance(window);
  volkLoadInstance(instance);

  fmt::print("Hello {} instance \n", "Vulkan");

  while (!window.should_close()) {
    window.poll_events();
    window.swap_buffers();
  }
}
