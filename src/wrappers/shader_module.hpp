#pragma once

#ifndef VULKAN_RENDERER_SHADER_MODULE_HPP
#define VULKAN_RENDERER_SHADER_MODULE_HPP

#include <vulkan/vulkan_core.h>

#include <string_view>
#include <vector>

#include "unique_resource.hpp"

struct UniqueShaderModule : UniqueResource<VkShaderModule> {
  UniqueShaderModule(
      VkDevice device, VkShaderModule resource,
      const VkAllocationCallbacks* allocator_ptr = nullptr) noexcept
      : UniqueResource<VkShaderModule>{device, resource, vkDestroyShaderModule,
                                       allocator_ptr}
  {
  }
};

[[nodiscard]] auto create_shader_module(VkDevice device,
                                        std::string_view filename)
    -> VkShaderModule;

[[nodiscard]] auto create_unique_shader_module(VkDevice device,
                                               std::string_view filename)
    -> UniqueShaderModule;

#endif // VULKAN_RENDERER_SHADER_MODULE_HPP
