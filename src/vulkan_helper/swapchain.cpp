#include "swapchain.hpp"
#include "gpu_device.hpp"

#include <beyond/utils/panic.hpp>

#include <VkBootstrap.h>

#include <vector>

namespace vkh {

Swapchain::Swapchain(const GPUDevice& device) : device_{device.device()}
{
  vkb::SwapchainBuilder swapchain_builder{
      device.vk_physical_device(), device.device(), device.surface(),
      device.queue_family_indices().graphics_family,
      device.queue_family_indices().present_family};

  auto swap_ret = swapchain_builder.use_default_format_selection()
                      .set_desired_format(VkSurfaceFormatKHR{
                          .format = VK_FORMAT_B8G8R8A8_SRGB,
                          .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                      })
                      .use_default_present_mode_selection()
                      .build();

  if (!swap_ret) { beyond::panic(vkb::to_string(swap_ret.error().type)); }

  swapchain_ = swap_ret->swapchain;
  images_count_ = swap_ret->image_count;
  images_ = vkb::get_swapchain_images(*swap_ret).value();
  image_views_ = vkb::get_swapchain_image_views(*swap_ret, images_).value();
  images_format_ = swap_ret->image_format;
  extent_ = swap_ret->extent;
}

Swapchain::~Swapchain()
{
  reset();
}

void Swapchain::reset() noexcept
{
  if (swapchain_ != nullptr) {
    for (auto* view : image_views_) {
      vkDestroyImageView(device_, view, nullptr);
    }
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
  }
  device_ = nullptr;
  swapchain_ = nullptr;
  images_.clear();
  image_views_.clear();
  images_format_ = VK_FORMAT_UNDEFINED;
  extent_ = {};
}

Swapchain::Swapchain(Swapchain&& other) noexcept
    : device_{std::exchange(other.device_, nullptr)},
      swapchain_{std::exchange(other.swapchain_, nullptr)}, images_{std::move(
                                                                other.images_)},
      image_views_{std::move(other.image_views_)},
      images_format_{std::exchange(other.images_format_, VK_FORMAT_UNDEFINED)},
      extent_{std::exchange(other.extent_, VkExtent2D{0, 0})}
{
}

auto Swapchain::operator=(Swapchain&& other) & noexcept -> Swapchain&
{
  device_ = std::exchange(other.device_, nullptr);
  swapchain_ = std::exchange(other.swapchain_, nullptr);
  images_ = std::move(other.images_);
  image_views_ = std::move(other.image_views_);
  images_format_ = std::exchange(other.images_format_, VK_FORMAT_UNDEFINED);
  extent_ = std::exchange(other.extent_, VkExtent2D{0, 0});
  return *this;
}

} // namespace vkh