#pragma once

#ifndef VULKAN_HELPER_INSTANCE_HPP
#define VULKAN_HELPER_INSTANCE_HPP

#include <array>
#include <vector>

namespace vkh {

[[nodiscard]] auto
create_instance(const char* title,
                std::vector<const char*> required_extensions) noexcept
    -> VkInstance;

#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
[[maybe_unused]] constexpr std::array validation_layers{
    "VK_LAYER_KHRONOS_validation"};

[[nodiscard]] auto create_debug_messenger(VkInstance instance) noexcept
    -> VkDebugUtilsMessengerEXT;
#endif

} // namespace vkh

#endif // VULKAN_HELPER_INSTANCE_HPP
