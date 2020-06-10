#ifndef VULKAN_RENDERER_TEXTURE_HPP
#define VULKAN_RENDERER_TEXTURE_HPP

#include "vulkan_helper/gpu_device.hpp"
#include "vulkan_helper/image.hpp"

#include <vulkan/vulkan.h>

struct Texture {
  uint32_t mip_levels{};
  VkImage image{};
  VmaAllocation image_allocation{};
  VkImageView image_view{};
  VkSampler sampler{};
};

[[nodiscard]] auto create_texture(vkh::GPUDevice& device, void* data,
                                  int tex_width, int tex_height) -> Texture;

[[nodiscard]] auto create_texture_from_file(vkh::GPUDevice& device,
                                            const char* texture_path)
    -> Texture;

auto destroy_texture(vkh::GPUDevice& device, Texture& texture) -> void;

#endif // VULKAN_RENDERER_TEXTURE_HPP
