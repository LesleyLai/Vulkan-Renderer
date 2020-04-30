#include "buffer.hpp"

namespace vkh {

auto create_buffer(VmaAllocator allocator, VkDeviceSize size,
                   VkBufferUsageFlags usage, VmaMemoryUsage memory_usage)
    -> beyond::expected<VmaCreateBufferResult, VkResult>
{
  const VkBufferCreateInfo buffer_create_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  const VmaAllocationCreateInfo allocation_create_info = {.usage =
                                                              memory_usage};

  VkBuffer buffer = nullptr;
  VmaAllocation allocation = nullptr;
  if (const auto result = vmaCreateBuffer(allocator, &buffer_create_info,
                                          &allocation_create_info, &buffer,
                                          &allocation, nullptr);
      result != VK_SUCCESS) {
    return beyond::unexpected{result};
  } else {
    return VmaCreateBufferResult{buffer, allocation};
  }
}

auto create_unique_buffer(VmaAllocator allocator, VkDeviceSize size,
                          VkBufferUsageFlags usage, VmaMemoryUsage memory_usage)
    -> beyond::expected<UniqueBuffer, VkResult>
{
  return create_buffer(allocator, size, usage, memory_usage)
      .map([&allocator](VmaCreateBufferResult result) {
        return UniqueBuffer(allocator, result.buffer, result.allocation);
      });
}

} // namespace vkh