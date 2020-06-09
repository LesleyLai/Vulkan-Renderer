#include "image_view.hpp"

namespace vkh {

[[nodiscard]] auto
create_image_view(VkDevice device,
                  const ImageViewCreateInfo& image_view_create_info)
    -> beyond::expected<VkImageView, VkResult>
{
  const VkImageViewCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image_view_create_info.image,
      .viewType = image_view_create_info.view_type,
      .format = image_view_create_info.format,
      .subresourceRange = image_view_create_info.subresource_range};

  VkImageView image_view = nullptr;
  if (auto res = vkCreateImageView(device, &create_info, nullptr, &image_view);
      res != VK_SUCCESS) {
    return beyond::unexpected(res);
  }

  return image_view;
}

[[nodiscard]] auto
create_unique_image_view(VkDevice device,
                         const ImageViewCreateInfo& image_view_create_info)
    -> beyond::expected<UniqueImageView, VkResult>
{
  return create_image_view(device, image_view_create_info)
      .map([&](VkImageView image_view) {
        return UniqueImageView(device, image_view);
      });
}

} // namespace vkh