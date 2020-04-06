#ifndef VULKAN_RENDERER_GPU_DEVICE_HPP
#define VULKAN_RENDERER_GPU_DEVICE_HPP

// #include <volk.h>

#include <vulkan/vulkan.h>

#include <vk_mem_alloc.h>

#include "queue_indices.hpp"

class Window;

enum EnableValidationLayer { yes = 0, no = 1 };

class GPUDevice {
  VkInstance instance_;

  EnableValidationLayer validation_layer_enabled_;

  VkDebugUtilsMessengerEXT debug_messenger_;

  VkPhysicalDevice physical_device_;
  VkSurfaceKHR surface_;
  VkDevice device_;

  QueueFamilyIndices queue_family_indices_;

  VkQueue graphics_queue_;
  VkQueue compute_queue_;
  VkQueue present_queue_;

  VmaAllocator allocator_;

public:
  explicit GPUDevice(Window& window,
                     EnableValidationLayer enable_validation_layer) noexcept;
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
      -> const QueueFamilyIndices&
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
