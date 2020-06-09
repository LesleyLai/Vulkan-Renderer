#include "buffer.hpp"
#include "single_time_command.hpp"

namespace vkh {

auto create_buffer(VmaAllocator allocator, VkDeviceSize size,
                   VkBufferUsageFlags usage, VmaMemoryUsage memory_usage)
    -> beyond::expected<VmaCreateBufferOutput, VkResult>
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
    return VmaCreateBufferOutput{buffer, allocation};
  }
}

auto create_unique_buffer(VmaAllocator allocator, VkDeviceSize size,
                          VkBufferUsageFlags usage, VmaMemoryUsage memory_usage)
    -> beyond::expected<UniqueBuffer, VkResult>
{
  return create_buffer(allocator, size, usage, memory_usage)
      .map([&allocator](VmaCreateBufferOutput result) {
        return UniqueBuffer(allocator, result.buffer, result.allocation);
      });
}

void copy_buffer(GPUDevice& device, VkBuffer src_buffer, VkBuffer dst_buffer,
                 VkDeviceSize size)
{
  vkh::execute_single_time_command(
      device.device(), device.graphics_command_pool(), device.graphics_queue(),
      [&](VkCommandBuffer command_buffer) {
        VkBufferCopy copyRegion = {.size = size};
        vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copyRegion);
      });
}

} // namespace vkh