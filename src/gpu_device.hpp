#ifndef VULKAN_RENDERER_GPU_DEVICE_HPP
#define VULKAN_RENDERER_GPU_DEVICE_HPP

#include <volk.h>

#include <vk_mem_alloc.h>

#include "vulkan_helper/queue_indices.hpp"

class Window;

class GPUDevice {
  VkInstance instance_;

#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
  VkDebugUtilsMessengerEXT debug_messenger_;
#endif

  VkPhysicalDevice physical_device_;
  VkSurfaceKHR surface_;
  VkDevice device_;

  vkh::QueueFamilyIndices queue_family_indices_;

  VkQueue graphics_queue_;
  VkQueue compute_queue_;
  VkQueue present_queue_;

  VmaAllocator allocator_;

public:
  explicit GPUDevice(Window& window) noexcept;
  ~GPUDevice() noexcept;

  GPUDevice(const GPUDevice&) = delete;
  auto operator=(const GPUDevice&) & -> GPUDevice& = delete;

  GPUDevice(GPUDevice&&) noexcept = delete;
  auto operator=(GPUDevice&&) & noexcept -> GPUDevice& = delete;

  [[nodiscard]] auto vk_instance() const noexcept -> VkInstance
  {
    return instance_;
  }

  [[nodiscard]] auto vk_physical_device() const noexcept -> VkPhysicalDevice
  {
    return physical_device_;
  }

  [[nodiscard]] auto vk_device() const noexcept -> VkDevice
  {
    return device_;
  }

  [[nodiscard]] auto surface() const noexcept -> VkSurfaceKHR
  {
    return surface_;
  }

  [[nodiscard]] auto queue_family_indices() const noexcept
      -> const vkh::QueueFamilyIndices&
  {
    return queue_family_indices_;
  }

  [[nodiscard]] auto graphics_queue() const noexcept -> VkQueue
  {
    return graphics_queue_;
  }

  [[nodiscard]] auto compute_queue() const noexcept -> VkQueue
  {
    return compute_queue_;
  }

  [[nodiscard]] auto present_queue() const noexcept -> VkQueue
  {
    return present_queue_;
  }

  [[nodiscard]] auto allocator() const noexcept -> VmaAllocator
  {
    return allocator_;
  }
};

#endif // VULKAN_RENDERER_GPU_DEVICE_HPP
