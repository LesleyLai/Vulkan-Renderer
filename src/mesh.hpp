#ifndef VULKAN_RENDERER_MESH_HPP
#define VULKAN_RENDERER_MESH_HPP

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#include <array>
#include <vulkan/vulkan_core.h>

#include "vulkan_helper/buffer.hpp"

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 tex_coord;

  [[nodiscard]] static constexpr auto get_binding_description() noexcept
      -> VkVertexInputBindingDescription
  {
    return {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
  }

  [[nodiscard]] static constexpr auto get_attribute_descriptions() noexcept
      -> std::array<VkVertexInputAttributeDescription, 3>
  {
    return {VkVertexInputAttributeDescription{
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, pos),
            },
            VkVertexInputAttributeDescription{
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, color),
            },
            VkVertexInputAttributeDescription{
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, tex_coord),
            }};
  }

  [[nodiscard]] auto operator==(const Vertex& other) const noexcept -> bool
  {
    return pos == other.pos && color == other.color &&
           tex_coord == other.tex_coord;
  }
};

struct StaticMesh {
  vkh::UniqueBuffer vertex_buffer;
  vkh::UniqueBuffer index_buffer;
  std::uint32_t indices_size;
};

[[nodiscard]] auto load_mesh(vkh::GPUDevice& device, VkCommandPool command_pool,
                             VkQueue queue, const char* path) -> StaticMesh;

#endif // VULKAN_RENDERER_MESH_HPP
