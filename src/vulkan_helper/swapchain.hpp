#pragma once

#ifndef VULKAN_HELPER_SWAPCHAIN_HPP
#define VULKAN_HELPER_SWAPCHAIN_HPP

#include <volk.h>

#include <optional>
#include <set>
#include <vector>

#include "queue_indices.hpp"

namespace vkh {

struct SwapchainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

[[nodiscard]] auto query_swapchain_support(VkPhysicalDevice device,
                                           VkSurfaceKHR surface) noexcept
    -> SwapchainSupportDetails;

class VulkanSwapchain {
public:
  VulkanSwapchain(VkPhysicalDevice pd, VkDevice device, VkSurfaceKHR surface,
                  const QueueFamilyIndices& indices);
  ~VulkanSwapchain();

  VulkanSwapchain(const VulkanSwapchain&) = delete;
  auto operator=(const VulkanSwapchain&) -> VulkanSwapchain& = delete;

  VulkanSwapchain(VulkanSwapchain&& other) noexcept
      : device_{std::exchange(other.device_, nullptr)},
        swapchain_{std::exchange(other.swapchain_, nullptr)},
        swapchain_images_{std::move(other.swapchain_images_)},
        swapchain_image_views_{std::move(other.swapchain_image_views_)},
        swapchain_images_format_{
            std::exchange(other.swapchain_images_format_, VK_FORMAT_UNDEFINED)},
        swapchain_extent_{
            std::exchange(other.swapchain_extent_, VkExtent2D{0, 0})}
  {
  }

  auto operator=(VulkanSwapchain&& other) & noexcept -> VulkanSwapchain&
  {
    device_ = std::exchange(other.device_, nullptr);
    swapchain_ = std::exchange(other.swapchain_, nullptr);
    swapchain_images_ = std::move(other.swapchain_images_);
    swapchain_image_views_ = std::move(other.swapchain_image_views_);
    swapchain_images_format_ =
        std::exchange(other.swapchain_images_format_, VK_FORMAT_UNDEFINED);
    swapchain_extent_ =
        std::exchange(other.swapchain_extent_, VkExtent2D{0, 0});
    return *this;
  }

private:
  VkDevice device_ = nullptr;
  VkSwapchainKHR swapchain_ = nullptr;
  std::vector<VkImage> swapchain_images_;
  std::vector<VkImageView> swapchain_image_views_;

  VkFormat swapchain_images_format_ = VK_FORMAT_UNDEFINED;
  VkExtent2D swapchain_extent_{};
};

} // namespace vkh

#endif // VULKAN_HELPER_SWAPCHAIN_HPP
