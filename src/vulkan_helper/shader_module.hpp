#pragma once

#ifndef VULKAN_HELPER_SHADER_MODULE_HPP
#define VULKAN_HELPER_SHADER_MODULE_HPP

#include <beyond/types/expected.hpp>

#include <vulkan/vulkan_core.h>

#include <string_view>
#include <vector>

#include "unique_resource.hpp"

namespace vkh {

struct UniqueShaderModule
    : UniqueResource<VkShaderModule, &vkDestroyShaderModule> {
  UniqueShaderModule(
      VkDevice device, VkShaderModule resource,
      const VkAllocationCallbacks* allocator_ptr = nullptr) noexcept
      : UniqueResource<VkShaderModule, &vkDestroyShaderModule>{device, resource,
                                                               allocator_ptr}
  {
  }
};

[[nodiscard]] auto create_shader_module(VkDevice device,
                                        std::string_view filename)
    -> beyond::expected<VkShaderModule, VkResult>;

[[nodiscard]] auto create_unique_shader_module(VkDevice device,
                                               std::string_view filename)
    -> beyond::expected<UniqueShaderModule, VkResult>;

} // namespace vkh

#endif // VULKAN_HELPER_SHADER_MODULE_HPP
