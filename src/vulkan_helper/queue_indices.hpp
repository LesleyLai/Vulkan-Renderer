#pragma once

#ifndef VULKAN_HELPER_QUEUE_INDICES_HPP
#define VULKAN_HELPER_QUEUE_INDICES_HPP

#include <optional>
#include <set>
#include <vector>
#include <vulkan/vulkan.h>

struct QueueFamilyIndices {
  std::uint32_t graphics_family;
  std::uint32_t present_family;
  std::uint32_t compute_family;

  [[nodiscard]] auto to_set() const noexcept -> std::set<std::uint32_t>
  {
    return std::set{graphics_family, present_family, compute_family};
  }
};

auto find_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface) noexcept
    -> std::optional<QueueFamilyIndices>;

#endif // VULKAN_HELPER_QUEUE_INDICES_HPP
