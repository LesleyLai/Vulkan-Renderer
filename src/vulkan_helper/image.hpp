#ifndef VULKAN_HELPER_IMAGE_HPP
#define VULKAN_HELPER_IMAGE_HPP

#include <vulkan/vulkan_core.h>

#include <beyond/types/expected.hpp>
#include <vk_mem_alloc.h>

#include <functional>

#include "gpu_device.hpp"

namespace vkh {

class UniqueImage {
  VmaAllocator allocator_{nullptr};
  VkImage image_{nullptr};
  VmaAllocation allocation_{nullptr};

public:
  UniqueImage() noexcept = default;
  UniqueImage(VmaAllocator allocator, VkImage image,
              VmaAllocation allocation) noexcept
      : allocator_{allocator}, image_{image}, allocation_{allocation}
  {
  }

  UniqueImage(const UniqueImage&) = delete;
  auto operator=(const UniqueImage&) -> UniqueImage& = delete;

  UniqueImage(UniqueImage&& other) noexcept
      : allocator_(other.allocator_),
        image_(std::exchange(other.image_, nullptr)),
        allocation_(std::exchange(other.allocation_, nullptr))
  {
  }

  auto operator=(UniqueImage&& other) & noexcept -> UniqueImage&
  {
    if (this != &other) {
      allocator_ = other.allocator_;
      image_ = std::exchange(other.image_, nullptr);
      allocation_ = std::exchange(other.allocation_, nullptr);
    }
    return *this;
  }

  auto reset() noexcept -> void
  {
    vmaDestroyImage(allocator_, image_, allocation_);
    image_ = nullptr;
    allocation_ = nullptr;
  }

  ~UniqueImage() noexcept
  {
    vmaDestroyImage(allocator_, image_, allocation_);
  }

  [[nodiscard]] auto get() const noexcept -> VkImage
  {
    return image_;
  }

  auto map(void** data) noexcept -> void
  {
    vmaMapMemory(allocator_, allocation_, data);
  }

  auto unmap() noexcept -> void
  {
    vmaUnmapMemory(allocator_, allocation_);
  }
};

struct VmaCreateImageOutput {
  VkImage image = nullptr;
  VmaAllocation allocation = nullptr;

  /*implicit*/ operator std::tuple<VkImage&, VmaAllocation&>()
  {
    return {image, allocation};
  }
};

struct ImageCreateInfo {
  VkExtent3D extent;
  uint32_t mip_levels = 1;
  VkSampleCountFlagBits samples_count = VK_SAMPLE_COUNT_1_BIT;
  VkFormat format;
  VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
  VkImageUsageFlags usage;
};

auto create_image(VmaAllocator allocator, const ImageCreateInfo& create_info,
                  VmaMemoryUsage memory_usage)
    -> beyond::expected<VmaCreateImageOutput, VkResult>;

auto create_unique_image(VmaAllocator allocator,
                         const ImageCreateInfo& create_info,
                         VmaMemoryUsage memory_usage)
    -> beyond::expected<UniqueImage, VkResult>;

} // namespace vkh

#endif // VULKAN_HELPER_IMAGE_HPP
