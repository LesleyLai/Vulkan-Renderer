#pragma once

#ifndef VULKAN_HELPER_SHADER_MODULE_HPP
#define VULKAN_HELPER_SHADER_MODULE_HPP

#include <volk.h>

#include <string_view>
#include <vector>

#include "unique_resource.hpp"

namespace vkh {

struct UniqueShaderModule : UniqueResource<VkShaderModule> {
  UniqueShaderModule(
      VkDevice device, VkShaderModule resource,
      const VkAllocationCallbacks* allocator_ptr = nullptr) noexcept
      : UniqueResource<VkShaderModule>{device, resource, vkDestroyShaderModule,
                                       allocator_ptr}
  {
  }
};

[[nodiscard]] auto create_shader_module(std::string_view filename,
                                        VkDevice device) -> VkShaderModule;

[[nodiscard]] auto create_unique_shader_module(std::string_view filename,
                                               VkDevice device)
    -> UniqueShaderModule;

} // namespace vkh

#endif // VULKAN_HELPER_SHADER_MODULE_HPP
