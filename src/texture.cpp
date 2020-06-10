#include "texture.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <beyond/utils/panic.hpp>

#include "vulkan_helper/buffer.hpp"
#include "vulkan_helper/image_view.hpp"
#include "vulkan_helper/single_time_command.hpp"

void transition_image_layout(vkh::GPUDevice& device, VkImage image,
                             VkFormat /*format*/, VkImageLayout old_layout,
                             VkImageLayout new_layout, uint32_t mip_levels)
{
  vkh::execute_single_time_command(
      device.device(), device.graphics_command_pool(), device.graphics_queue(),
      [&](VkCommandBuffer command_buffer) {
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = mip_levels,
                .baseArrayLayer = 0,
                .layerCount = 1,
            }};

        VkPipelineStageFlags source_stage;
        VkPipelineStageFlags destination_stage;

        if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
            new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
          barrier.srcAccessMask = 0;
          barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

          source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
          destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
          barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

          source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
          destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
          beyond::panic("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
      });
}

void copy_buffer_to_image(vkh::GPUDevice& device, VkBuffer buffer,
                          VkImage image, uint32_t width, uint32_t height)
{
  vkh::execute_single_time_command(
      device.device(), device.graphics_command_pool(), device.graphics_queue(),
      [&](VkCommandBuffer command_buffer) {
        const VkBufferImageCopy region = {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1},
        };

        vkCmdCopyBufferToImage(command_buffer, buffer, image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);
      });
}

void generate_mipmaps(vkh::GPUDevice& device, VkImage image,
                      VkFormat imageFormat, int32_t tex_width,
                      int32_t tex_height, uint32_t mip_levels)
{
  // Check if image format supports linear blitting
  VkFormatProperties formatProperties;
  vkGetPhysicalDeviceFormatProperties(device.vk_physical_device(), imageFormat,
                                      &formatProperties);

  if (!(formatProperties.optimalTilingFeatures &
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
    beyond::panic("texture image format does not support linear blitting!");
  }

  vkh::execute_single_time_command(
      device.device(), device.graphics_command_pool(), device.graphics_queue(),
      [&](VkCommandBuffer command_buffer) {
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            }};

        int32_t mip_width = tex_width;
        int32_t mip_height = tex_height;

        for (uint32_t i = 1; i < mip_levels; ++i) {
          barrier.subresourceRange.baseMipLevel = i - 1;
          barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
          barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
          barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

          vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                               nullptr, 1, &barrier);

          const VkImageBlit blit = {
              .srcSubresource{
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .mipLevel = i - 1,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
              .srcOffsets = {{0, 0, 0}, {mip_width, mip_height, 1}},
              .dstSubresource{
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .mipLevel = i,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
              .dstOffsets = {{0, 0, 0},
                             {mip_width > 1 ? mip_width / 2 : 1,
                              mip_height > 1 ? mip_height / 2 : 1, 1}}};

          vkCmdBlitImage(command_buffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                         VK_FILTER_LINEAR);

          barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
          barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
          barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

          vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                               nullptr, 0, nullptr, 1, &barrier);

          if (mip_width > 1) mip_width /= 2;
          if (mip_height > 1) mip_height /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mip_levels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
      });
}

[[nodiscard]] auto create_texture(vkh::GPUDevice& device, void* data,
                                  int tex_width, int tex_height) -> Texture
{
  // TODO: error handling

  Texture texture;

  const auto image_size = static_cast<VkDeviceSize>(tex_width * tex_height * 4);

  texture.mip_levels = static_cast<uint32_t>(std::floor(
                           std::log2(std::max(tex_width, tex_height)))) +
                       1;

  auto staging_buffer =
      vkh::create_unique_buffer(device.allocator(), image_size,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VMA_MEMORY_USAGE_CPU_TO_GPU)
          .value(); // TODO: Better error message

  void* staging_ptr = nullptr;
  staging_buffer.map(&staging_ptr);
  memcpy(staging_ptr, data, image_size);
  staging_buffer.unmap();

  auto texture_image_res = vkh::create_image(
      device.allocator(),
      vkh::ImageCreateInfo{
          .extent = {static_cast<uint32_t>(tex_width),
                     static_cast<uint32_t>(tex_height), 1},
          .mip_levels = texture.mip_levels,
          .samples_count = VK_SAMPLE_COUNT_1_BIT,
          .format = VK_FORMAT_R8G8B8A8_SRGB,
          .tiling = VK_IMAGE_TILING_OPTIMAL,
          .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      },
      VMA_MEMORY_USAGE_GPU_ONLY);
  std::tie(texture.image, texture.image_allocation) =
      texture_image_res.value(); // TODO: Better error message

  transition_image_layout(
      device, texture.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, texture.mip_levels);
  copy_buffer_to_image(device, staging_buffer.get(), texture.image,
                       static_cast<uint32_t>(tex_width),
                       static_cast<uint32_t>(tex_height));
  // transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating
  // mipmaps

  generate_mipmaps(device, texture.image, VK_FORMAT_R8G8B8A8_SRGB, tex_width,
                   tex_height, texture.mip_levels);

  texture.image_view =
      vkh::create_image_view(device.device(),
                             {.image = texture.image,
                              .view_type = VK_IMAGE_VIEW_TYPE_2D,
                              .format = VK_FORMAT_R8G8B8A8_SRGB,
                              .subresource_range =
                                  {
                                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .baseMipLevel = 0,
                                      .levelCount = texture.mip_levels,
                                      .baseArrayLayer = 0,
                                      .layerCount = 1,
                                  }})
          .value();
  // TODO: better error message

  const VkSamplerCreateInfo sampler_create_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .mipLodBias = 0,
      .anisotropyEnable = VK_TRUE,
      .maxAnisotropy = 16,
      .compareEnable = VK_FALSE,
      .compareOp = VK_COMPARE_OP_ALWAYS,
      .minLod = 0,
      .maxLod = static_cast<float>(texture.mip_levels),
      .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
      .unnormalizedCoordinates = VK_FALSE,
  };

  if (vkCreateSampler(device.device(), &sampler_create_info, nullptr,
                      &texture.sampler) != VK_SUCCESS) {
    beyond::panic("Cannot create texture image");
  };

  return texture;
}

[[nodiscard]] auto create_texture_from_file(vkh::GPUDevice& device,
                                            const char* texture_path) -> Texture
{
  int tex_width, tex_height, tex_channels;

  stbi_uc* pixels = stbi_load(texture_path, &tex_width, &tex_height,
                              &tex_channels, STBI_rgb_alpha);

  if (!pixels) {
    beyond::panic(
        fmt::format("failed to load texture image at {}!", texture_path));
  }

  Texture t = create_texture(device, pixels, tex_width, tex_height);

  stbi_image_free(pixels);

  return t;
}

auto destroy_texture(vkh::GPUDevice& device, Texture& texture) -> void
{
  vkDestroySampler(device.device(), texture.sampler, nullptr);
  vkDestroyImageView(device.device(), texture.image_view, nullptr);
  vmaDestroyImage(device.allocator(), texture.image, texture.image_allocation);
}