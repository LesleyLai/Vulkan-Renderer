#include <array>
#include <cstring>
#include <iostream>

#include "instance.hpp"
#include "panic.hpp"
#include "utils.hpp"

namespace {

#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
constexpr std::array validation_layers = {"VK_LAYER_KHRONOS_validation"};

auto check_validation_layer_support() noexcept -> bool
{
  const auto available = vkh::get_vector_with<VkLayerProperties>(
      vkEnumerateInstanceLayerProperties);

  return std::all_of(
      std::begin(validation_layers), std::end(validation_layers),
      [&](const char* layer_name) {
        return std::find_if(std::begin(available), std::end(available),
                            [&](const auto& layer_properties) {
                              return std::strcmp(
                                  layer_name, static_cast<const char*>(
                                                  layer_properties.layerName));
                            }) != std::end(available);
      });
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
               VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
               const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
               void* /*pUserData*/) noexcept
{
  std::cout << "validation layer: " << p_callback_data->pMessage << "\n";
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

} // anonymous namespace

namespace vkh {

[[nodiscard]] auto
create_instance(const char* title,
                std::vector<const char*> required_extensions) noexcept
    -> VkInstance
{
#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
  if (!check_validation_layer_support()) {
    vkh::panic("validation layers requested, but not available!");
  }
#endif

  VkInstance instance;

  const VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = nullptr,
      .pApplicationName = title,
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "Vulkan Renderer",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_2,
  };

#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
  required_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = vkh::to_u32(required_extensions.size());
  create_info.ppEnabledExtensionNames = required_extensions.data();

#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
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
    vkh::panic("Cannot create vulkan instance!");
  }

  volkLoadInstance(instance);

  return instance;
}

} // namespace vkh