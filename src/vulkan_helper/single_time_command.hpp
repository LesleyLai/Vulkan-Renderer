#ifndef VULKAN_HELPER_SINGLE_TIME_COMMAND_HPP
#define VULKAN_HELPER_SINGLE_TIME_COMMAND_HPP

#include <beyond/utils/function_ref.hpp>

#include "vulkan/vulkan_core.h"

namespace vkh {

/**
 * @brief Executes one-shot GPU command
 * @param device The Vulkan Device to execute the command on
 * @param command_pool The command to send the command
 * @param queue The queue to send the command
 * @param func A function that record the command
 */
auto execute_single_time_command(
    VkDevice device, VkCommandPool command_pool, VkQueue queue,
    beyond::function_ref<void(VkCommandBuffer)> func) -> void;

} // namespace vkh

#endif // VULKAN_HELPER_SINGLE_TIME_COMMAND_HPP
