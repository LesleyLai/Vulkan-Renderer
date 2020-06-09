#ifndef VULKAN_HELPER_GPU_DEVICE_HPP
#define VULKAN_HELPER_GPU_DEVICE_HPP

#include <cstdint>

#include <vulkan/vulkan.h>

#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

class Window;

namespace vkh {

enum ValidationLayerSetting { enable = 0, disable = 1 };

struct QueueFamilyIndices {
  std::uint32_t graphics_family;
  std::uint32_t present_family;
  std::uint32_t compute_family;
};

class GPUDevice {
  vkb::Instance instance_;
  VkPhysicalDevice physical_device_;
  VkSurfaceKHR surface_;
  VkDevice device_;

  VkSampleCountFlagBits msaa_sample_count_;

  VkQueue graphics_queue_;
  VkQueue compute_queue_;
  VkQueue present_queue_;

  QueueFamilyIndices queue_family_indices_;

  VmaAllocator allocator_{};

  VkCommandPool graphics_command_pool_;

public:
  explicit GPUDevice(Window& window,
                     ValidationLayerSetting validation_layer_setting =
                         ValidationLayerSetting::enable) noexcept;
  ~GPUDevice() noexcept;

  GPUDevice(const GPUDevice&) = delete;
  auto operator=(const GPUDevice&) & -> GPUDevice& = delete;

  GPUDevice(GPUDevice&&) noexcept = delete;
  auto operator=(GPUDevice&&) & noexcept -> GPUDevice& = delete;

  [[nodiscard]] auto vk_instance() const noexcept -> VkInstance
  {
    return instance_.instance;
  }

  [[nodiscard]] auto vk_physical_device() const noexcept -> VkPhysicalDevice
  {
    return physical_device_;
  }

  [[nodiscard]] auto device() const noexcept -> VkDevice
  {
    return device_;
  }

  [[nodiscard]] auto surface() const noexcept -> VkSurfaceKHR
  {
    return surface_;
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

  [[nodiscard]] auto msaa_sample_count() const noexcept -> VkSampleCountFlagBits
  {
    return msaa_sample_count_;
  }

  [[nodiscard]] auto queue_family_indices() const noexcept
      -> const QueueFamilyIndices&
  {
    return queue_family_indices_;
  }

  [[nodiscard]] auto graphics_command_pool() const noexcept -> VkCommandPool
  {
    return graphics_command_pool_;
  }

  auto wait_idle() const noexcept -> void;
};

} // namespace vkh

#endif // VULKAN_HELPER_GPU_DEVICE_HPP
