#ifndef VULKAN_RENDERER_BUFFER_HPP
#define VULKAN_RENDERER_BUFFER_HPP

#include <vulkan/vulkan_core.h>

#include <beyond/types/expected.hpp>
#include <vk_mem_alloc.h>

#include <functional>

#include "gpu_device.hpp"

namespace vkh {

class UniqueBuffer {
  VmaAllocator allocator_{nullptr};
  VkBuffer buffer_{nullptr};
  VmaAllocation allocation_{nullptr};

public:
  UniqueBuffer() noexcept = default;
  UniqueBuffer(VmaAllocator allocator, VkBuffer buffer,
               VmaAllocation allocation) noexcept
      : allocator_{allocator}, buffer_{buffer}, allocation_{allocation}
  {
  }

  UniqueBuffer(const UniqueBuffer&) = delete;
  auto operator=(const UniqueBuffer&) -> UniqueBuffer& = delete;

  UniqueBuffer(UniqueBuffer&& other) noexcept
      : allocator_(other.allocator_),
        buffer_(std::exchange(other.buffer_, nullptr)),
        allocation_(std::exchange(other.allocation_, nullptr))
  {
  }

  auto operator=(UniqueBuffer&& other) & noexcept -> UniqueBuffer&
  {
    allocator_ = other.allocator_;
    buffer_ = std::exchange(other.buffer_, nullptr);
    allocation_ = std::exchange(other.allocation_, nullptr);
    return *this;
  }

  auto reset() noexcept -> void
  {
    vmaDestroyBuffer(allocator_, buffer_, allocation_);
    buffer_ = nullptr;
    allocation_ = nullptr;
  }

  ~UniqueBuffer() noexcept
  {
    vmaDestroyBuffer(allocator_, buffer_, allocation_);
  }

  [[nodiscard]] /*implicit*/ operator VkBuffer() const noexcept
  {
    return buffer_;
  }

  [[nodiscard]] auto get() const noexcept -> VkBuffer
  {
    return buffer_;
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

struct VmaCreateBufferResult {
  VkBuffer buffer = nullptr;
  VmaAllocation allocation = nullptr;

  /*implicit*/ operator std::tuple<VkBuffer&, VmaAllocation&>()
  {
    return {buffer, allocation};
  }
};

auto create_buffer(VmaAllocator allocator, VkDeviceSize size,
                   VkBufferUsageFlags usage, VmaMemoryUsage memory_usage)
    -> beyond::expected<VmaCreateBufferResult, VkResult>;

auto create_unique_buffer(VmaAllocator allocator, VkDeviceSize size,
                          VkBufferUsageFlags usage, VmaMemoryUsage memory_usage)
    -> beyond::expected<UniqueBuffer, VkResult>;

} // namespace vkh

#endif // VULKAN_RENDERER_BUFFER_HPP
