#pragma once

#ifndef VULKAN_HELPER_SHADER_MODULE_HPP
#define VULKAN_HELPER_SHADER_MODULE_HPP

#include <volk.h>

#include <string_view>
#include <vector>

namespace vkh {

[[nodiscard]] auto create_shader_module(std::string_view filename,
                                        VkDevice device) -> VkShaderModule;

[[nodiscard]] auto create_shader_module(std::size_t size, const uint32_t*,
                                        VkDevice device) -> VkShaderModule;

} // namespace vkh

#endif // VULKAN_HELPER_SHADER_MODULE_HPP
