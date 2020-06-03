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

#include <gsl/span>

struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
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
                .offset = offsetof(Vertex, normal),
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
    return pos == other.pos && normal == other.normal &&
           tex_coord == other.tex_coord;
  }
};

struct StaticMesh {
  vkh::UniqueBuffer vertex_buffer;
  vkh::UniqueBuffer index_buffer;
  std::uint32_t indices_size;
};

[[nodiscard]] auto
create_mesh_from_data(vkh::GPUDevice& device, VkCommandPool command_pool,
                      VkQueue queue, gsl::span<Vertex> vertices,
                      gsl::span<uint32_t> indices) -> StaticMesh;

#endif // VULKAN_RENDERER_MESH_HPP
