#include <vector>

#include "mesh.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace std {
template <> struct hash<Vertex> {
  [[nodiscard]] auto operator()(Vertex const& vertex) const noexcept -> size_t
  {
    return ((hash<glm::vec3>()(vertex.pos) ^
             (hash<glm::vec3>()(vertex.normal) << 1u)) >>
            1u) ^
           (hash<glm::vec2>()(vertex.tex_coord) << 1u);
  }
};
} // namespace std

namespace {
auto create_vertex_buffer(vkh::GPUDevice& device, VkCommandPool command_pool,
                          VkQueue queue, gsl::span<Vertex> vertices)
    -> vkh::UniqueBuffer
{
  VkDeviceSize buffer_size =
      sizeof(vertices[0]) * static_cast<VkDeviceSize>(vertices.size());

  auto staging_buffer =
      vkh::create_unique_buffer(device.allocator(), buffer_size,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VMA_MEMORY_USAGE_CPU_TO_GPU)
          .value();

  void* data = nullptr;
  staging_buffer.map(&data);
  memcpy(data, vertices.data(), buffer_size);
  staging_buffer.unmap();

  vkh::UniqueBuffer vertex_buffer =
      vkh::create_unique_buffer(device.allocator(), buffer_size,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY)
          .value();

  vkh::copy_buffer(device, command_pool, queue, staging_buffer, vertex_buffer,
                   buffer_size);

  return vertex_buffer;
}

auto create_index_buffer(vkh::GPUDevice& device, VkCommandPool command_pool,
                         VkQueue queue, gsl::span<uint32_t> indices)
    -> vkh::UniqueBuffer
{
  VkDeviceSize buffer_size =
      sizeof(indices[0]) * static_cast<VkDeviceSize>(indices.size());

  auto staging_buffer =
      vkh::create_unique_buffer(device.allocator(), buffer_size,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VMA_MEMORY_USAGE_CPU_TO_GPU)
          .value();

  void* data = nullptr;
  staging_buffer.map(&data);
  memcpy(data, indices.data(), buffer_size);
  staging_buffer.unmap();

  vkh::UniqueBuffer index_buffer =
      vkh::create_unique_buffer(device.allocator(), buffer_size,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY)
          .value();

  vkh::copy_buffer(device, command_pool, queue, staging_buffer, index_buffer,
                   buffer_size);

  return index_buffer;
}
} // namespace

[[nodiscard]] auto create_mesh_from_file(vkh::GPUDevice& device,
                                         VkCommandPool command_pool,
                                         VkQueue queue, const char* path)
    -> StaticMesh
{
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path)) {
    throw std::runtime_error(err);
  }

  std::unordered_map<Vertex, uint32_t> unique_vertices{};

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  for (const auto& shape : shapes) {
    for (const auto& index : shape.mesh.indices) {
      Vertex vertex = {
          .pos =
              {attrib.vertices[static_cast<size_t>(3 * index.vertex_index + 0)],
               attrib.vertices[static_cast<size_t>(3 * index.vertex_index + 1)],
               attrib
                   .vertices[static_cast<size_t>(3 * index.vertex_index + 2)]},
          .normal = {1.0f, 1.0f, 1.0f},
          .tex_coord = {attrib.texcoords[static_cast<size_t>(
                            2 * index.texcoord_index + 0)],
                        1.0f - attrib.texcoords[static_cast<size_t>(
                                   2 * index.texcoord_index + 1)]},
      };

      if (unique_vertices.count(vertex) == 0) {
        unique_vertices[vertex] = static_cast<uint32_t>(vertices.size());
        vertices.push_back(vertex);
      }

      indices.push_back(unique_vertices[vertex]);
    }
  }

  return create_mesh_from_data(device, command_pool, queue, vertices, indices);
}

[[nodiscard]] auto
create_mesh_from_data(vkh::GPUDevice& device, VkCommandPool command_pool,
                      VkQueue queue, gsl::span<Vertex> vertices,
                      gsl::span<uint32_t> indices) -> StaticMesh
{
  return StaticMesh{
      .vertex_buffer =
          create_vertex_buffer(device, command_pool, queue, vertices),
      .index_buffer = create_index_buffer(device, command_pool, queue, indices),
      .indices_size = static_cast<uint32_t>(indices.size())};
}