//#include "gpu_device.hpp"
//#include "window.hpp"
//
//#include <array>
//#include <cstring>
//#include <iostream>
//
//#include "../vulkan_helper/panic.hpp"
//#include "../vulkan_helper/utils.hpp"
//
//#include "VkBootstrap.h"
//
// namespace {
//
//[[maybe_unused]] constexpr std::array validation_layers{
//    "VK_LAYER_KHRONOS_validation"};
//
//// auto check_validation_layer_support() noexcept -> bool
////{
////  const auto available = vkh::get_vector_with<VkLayerProperties>(
////      vkEnumerateInstanceLayerProperties);
////
////  return std::all_of(
////      std::begin(validation_layers), std::end(validation_layers),
////      [&](const char* layer_name) {
////        return std::find_if(std::begin(available), std::end(available),
////                            [&](const auto& layer_properties) {
////                              return std::strcmp(
////                                  layer_name, static_cast<const char*>(
//// layer_properties.layerName)); /                            }) !=
/// std::end(available); /      });
////}
//
// VKAPI_ATTR VkBool32 VKAPI_CALL
// debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
//               VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
//               const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
//               void* /*pUserData*/) noexcept
//{
//  std::cout << "validation layer: " << p_callback_data->pMessage << "\n";
//  return VK_FALSE;
//}
//
// constexpr auto populate_debug_messenger_create_info() noexcept
//    -> VkDebugUtilsMessengerCreateInfoEXT
//{
//  return {
//      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
//      .pNext = nullptr,
//      .flags = 0,
//      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
//                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
//                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
//      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
//                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
//                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
//      .pfnUserCallback = debug_callback,
//      .pUserData = nullptr,
//  };
//}
//
//} // anonymous namespace
//
//[[nodiscard]] auto
// create_instance(const char* title,
//                std::vector<const char*> required_extensions) noexcept
//    -> VkInstance
//{
//#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
//  if (!check_validation_layer_support()) {
//    vkh::panic("validation layers requested, but not available!");
//  }
//#endif
//
//  VkInstance instance;
//
//  const VkApplicationInfo app_info = {
//      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
//      .pNext = nullptr,
//      .pApplicationName = title,
//      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
//      .pEngineName = "Vulkan Renderer",
//      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
//      .apiVersion = VK_API_VERSION_1_2,
//  };
//
//#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
//  required_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
//#endif
//
//  VkInstanceCreateInfo create_info{};
//  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
//  create_info.pApplicationInfo = &app_info;
//  create_info.enabledExtensionCount = vkh::to_u32(required_extensions.size());
//  create_info.ppEnabledExtensionNames = required_extensions.data();
//
//#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
//  create_info.enabledLayerCount =
//      static_cast<uint32_t>(vkh::validation_layers.size());
//  create_info.ppEnabledLayerNames = vkh::validation_layers.data();
//
//  VkDebugUtilsMessengerCreateInfoEXT debug_create_info =
//      populate_debug_messenger_create_info();
//  create_info.pNext = &debug_create_info;
//#else
//  create_info.enabledLayerCount = 0;
//  create_info.pNext = nullptr;
//#endif
//
//  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
//    //vkh::panic("Cannot create vulkan instance!");
//  }
//
//  return instance;
//}
//
//[[nodiscard]] auto create_debug_messenger(VkInstance instance) noexcept
//    -> VkDebugUtilsMessengerEXT
//{
//
//  const VkDebugUtilsMessengerCreateInfoEXT create_info =
//      populate_debug_messenger_create_info();
//
//  VkDebugUtilsMessengerEXT debug_mesenger;
//  auto result = vkCreateDebugUtilsMessengerEXT(instance, &create_info,
//  nullptr,
//                                               &debug_mesenger);
//  if (result != VK_SUCCESS) {
//    vkh::panic("failed to set up debug messenger!");
//  }
//  return debug_mesenger;
//}
//
//#include <beyond/core/utils/panic.hpp>
//
// GPUDevice::GPUDevice(Window& window,
//                     EnableValidationLayer enable_validation_layer) noexcept
//{
//  instance_ = create_instance(window.title().c_str(),
//                              window.get_required_instance_extensions());
//
//  window.create_vulkan_surface(instance_, nullptr, surface_);
//
//  validation_layer_enabled_ = enable_validation_layer;
//
//  if (validation_layer_enabled_ == EnableValidationLayer::yes) {
//    debug_messenger_ = create_debug_messenger(instance_);
//  }
//
//  //  physical_device_ = vkh::pick_physical_device(instance_, surface_);
//  //  queue_family_indices_ = [&]() {
//  //    auto indices = vkh::find_queue_families(physical_device_, surface_);
//  //    if (!indices) {
//  //      beyond::panic("Cannot find a physical device that satisfy all the
//  //      queue
//  //                    "
//  //                    "family indices requirements");
//  //    }
//  //    return *indices;
//  //  }();
//  //
//  //  device_ = vkh::create_logical_device(physical_device_,
//  //  queue_family_indices_); volkLoadDevice(device_);
//  //
//  //  const auto get_device_queue = [this](std::uint32_t family_index,
//  //                                       std::uint32_t index) {
//  //    VkQueue queue;
//  //    vkGetDeviceQueue(device_, family_index, index, &queue);
//  //    return queue;
//  //  };
//  //  graphics_queue_ =
//  get_device_queue(queue_family_indices_.graphics_family,
//  //  0); present_queue_ =
//  //  get_device_queue(queue_family_indices_.present_family, 0);
//  // compute_queue_ = get_device_queue(queue_family_indices_.compute_family,
//  0);
//
//  //  VmaAllocatorCreateInfo allocator_info{};
//  //  allocator_info.physicalDevice = physical_device_;
//  //  allocator_info.device = device_;
//  //
//  //  VKH_CHECK(vmaCreateAllocator(&allocator_info, &allocator_));
//}
//
// GPUDevice::~GPUDevice() noexcept
//{
//  // vmaDestroyAllocator(allocator_);
//  //  vkDestroyDevice(device_, nullptr);
//  //#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
//  //  vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
//  //#endif
//  //  vkDestroySurfaceKHR(instance_, surface_, nullptr);
//  vkDestroyInstance(instance_, nullptr);
//}
