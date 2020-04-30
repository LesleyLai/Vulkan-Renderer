#include "single_time_command.hpp"

auto vkh::execute_single_time_command(
    VkDevice device, VkCommandPool command_pool, VkQueue queue,
    beyond::function_ref<void(VkCommandBuffer)> func) -> void
{
  const VkCommandBufferAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };

  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);

  const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

  vkBeginCommandBuffer(command_buffer, &begin_info);

  func(command_buffer);

  vkEndCommandBuffer(command_buffer);

  const VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
  };

  vkQueueSubmit(queue, 1, &submit_info, nullptr);
  vkQueueWaitIdle(queue);

  vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}
