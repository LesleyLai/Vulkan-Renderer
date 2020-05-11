#include "image.hpp"

namespace vkh {

auto create_image(VmaAllocator allocator, const ImageCreateInfo& create_info,
                  VmaMemoryUsage memory_usage)
    -> beyond::expected<VmaCreateImageOutput, VkResult>
{

  const VkImageCreateInfo vk_image_create_info{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = create_info.format,
      .extent = create_info.extent,
      .mipLevels = create_info.mip_levels,
      .arrayLayers = 1,
      .samples = create_info.samples_count,
      .tiling = create_info.tiling,
      .usage = create_info.usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  VmaAllocationCreateInfo allocation_create_info{.usage = memory_usage};

  VmaCreateImageOutput output;
  if (VkResult result = vmaCreateImage(allocator, &vk_image_create_info,
                                       &allocation_create_info, &output.image,
                                       &output.allocation, nullptr);
      !result) {
    return beyond::unexpected{result};
  } else {
    return output;
  }
}

auto create_unique_image(VmaAllocator allocator,
                         const ImageCreateInfo& create_info,
                         VmaMemoryUsage memory_usage)
    -> beyond::expected<UniqueImage, VkResult>
{
  return create_image(allocator, create_info, memory_usage)
      .transform([&](const VmaCreateImageOutput& output) {
        return UniqueImage(allocator, output.image, output.allocation);
      });
}

} // namespace vkh