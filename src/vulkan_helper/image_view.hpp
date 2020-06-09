#ifndef VULKAN_HELPER_IMAGE_VIEW_HPP
#define VULKAN_HELPER_IMAGE_VIEW_HPP

#include "unique_resource.hpp"

#include <beyond/types/expected.hpp>

namespace vkh {

struct ImageViewCreateInfo {
  VkImageViewCreateFlags flags;
  VkImage image;
  VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;
  VkFormat format;
  VkComponentMapping components{};
  VkImageSubresourceRange subresource_range;
};

struct UniqueImageView : UniqueResource<VkImageView, vkDestroyImageView> {
  using UniqueResource<VkImageView, vkDestroyImageView>::UniqueResource;
};

[[nodiscard]] auto
create_image_view(VkDevice device,
                  const ImageViewCreateInfo& image_view_create_info)
    -> beyond::expected<VkImageView, VkResult>;

[[nodiscard]] auto
create_unique_image_view(VkDevice device,
                         const ImageViewCreateInfo& image_view_create_info)
    -> beyond::expected<UniqueImageView, VkResult>;

} // namespace vkh

#endif // VULKAN_HELPER_IMAGE_VIEW_HPP
