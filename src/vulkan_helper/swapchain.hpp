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

  [[nodiscard]] auto image_count() const noexcept -> std::uint32_t
  {
    return images_count_;
  }

  [[nodiscard]] auto image_format() const noexcept -> VkFormat
  {
    return images_format_;
  }

  [[nodiscard]] auto images() const noexcept -> const std::vector<VkImage>&
  {
    return images_;
  }

  [[nodiscard]] auto image_views() const noexcept
      -> const std::vector<VkImageView>&
  {
    return image_views_;
  }

  [[nodiscard]] auto extent() const noexcept -> VkExtent2D
  {
    return extent_;
  }

private:
  VkDevice device_ = nullptr;
  VkSwapchainKHR swapchain_ = nullptr;
  std::uint32_t images_count_;
  std::vector<VkImage> images_;
  std::vector<VkImageView> image_views_;

  VkFormat images_format_ = VK_FORMAT_UNDEFINED;
  VkExtent2D extent_{};
};

} // namespace vkh

#endif // VULKAN_HELPER_SWAPCHAIN_HPP
