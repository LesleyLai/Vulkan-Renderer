//#include "gpu_device.hpp"
//
//#include "vulkan_helper/check.hpp"
//#include "vulkan_helper/instance.hpp"
//#include "vulkan_helper/logical_device.hpp"
//#include "vulkan_helper/physical_device_picker.hpp"
//#include "window.hpp"
//
//#include <beyond/core/utils/panic.hpp>
//
// GPUDevice::GPUDevice(Window& window) noexcept
//{
//  instance_ = vkh::create_instance(window.title().c_str(),
//                                   window.get_required_instance_extensions());
//
//  window.create_vulkan_surface(instance_, nullptr, surface_);
//
//#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
//  debug_messenger_ = vkh::create_debug_messenger(instance_);
//#endif
//
//  physical_device_ = vkh::pick_physical_device(instance_, surface_);
//  queue_family_indices_ = [&]() {
//    auto indices = vkh::find_queue_families(physical_device_, surface_);
//    if (!indices) {
//      beyond::panic("Cannot find a physical device that satisfy all the queue
//      "
//                    "family indices requirements");
//    }
//    return *indices;
//  }();
//
//  device_ = vkh::create_logical_device(physical_device_,
//  queue_family_indices_); volkLoadDevice(device_);
//
//  const auto get_device_queue = [this](std::uint32_t family_index,
//                                       std::uint32_t index) {
//    VkQueue queue;
//    vkGetDeviceQueue(device_, family_index, index, &queue);
//    return queue;
//  };
//  graphics_queue_ = get_device_queue(queue_family_indices_.graphics_family,
//  0); present_queue_ = get_device_queue(queue_family_indices_.present_family,
//  0); compute_queue_ = get_device_queue(queue_family_indices_.compute_family,
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
//  vkDestroyDevice(device_, nullptr);
//#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
//  vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
//#endif
//  vkDestroySurfaceKHR(instance_, surface_, nullptr);
//  vkDestroyInstance(instance_, nullptr);
//}
