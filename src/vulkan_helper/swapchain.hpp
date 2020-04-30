#pragma once

#ifndef VULKAN_HELPER_SWAPCHAIN_HPP
#define VULKAN_HELPER_SWAPCHAIN_HPP

#include <vulkan/vulkan.h>

#include <optional>
#include <set>
#include <vector>

namespace vkh {

struct SwapchainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

[[nodiscard]] auto query_swapchain_support(VkPhysicalDevice device,
                                           VkSurfaceKHR surface) noexcept
    -> SwapchainSupportDetails;

class GPUDevice;

class Swapchain {
public:
  Swapchain() = default;
  explicit Swapchain(const GPUDevice& device);
  ~Swapchain();

  Swapchain(const Swapchain&) = delete;
  auto operator=(const Swapchain&) -> Swapchain& = delete;

  Swapchain(Swapchain&& other) noexcept;

  auto operator=(Swapchain&& other) & noexcept -> Swapchain&;

  void reset() noexcept;

  [[nodiscard]] auto get() const noexcept -> VkSwapchainKHR
  {
    return swapchain_;
  }

  /// @brief Implicit convert to the underlying VkSwapchainKHR
  [[nodiscard]] /*implicit*/ operator VkSwapchainKHR() const noexcept
  {
    return swapchain_;
  }

  [[nodiscard]] auto image_format() const noexcept -> VkFormat
  {
    return swapchain_images_format_;
  }

  [[nodiscard]] auto images() const noexcept -> const std::vector<VkImage>&
  {
    return swapchain_images_;
  }

  [[nodiscard]] auto image_views() const noexcept
      -> const std::vector<VkImageView>&
  {
    return swapchain_image_views_;
  }

  [[nodiscard]] auto extent() const noexcept -> VkExtent2D
  {
    return swapchain_extent_;
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
