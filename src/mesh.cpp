#include <vector>

#include "mesh.hpp"

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
auto create_vertex_buffer(vkh::GPUDevice& device, gsl::span<Vertex> vertices)
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

  vkh::copy_buffer(device, staging_buffer.get(), vertex_buffer.get(),
                   buffer_size);

  return vertex_buffer;
}

auto create_index_buffer(vkh::GPUDevice& device, gsl::span<uint32_t> indices)
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

  vkh::copy_buffer(device, staging_buffer.get(), index_buffer.get(),
                   buffer_size);

  return index_buffer;
}
} // namespace

[[nodiscard]] auto create_mesh_from_data(vkh::GPUDevice& device,
                                         gsl::span<Vertex> vertices,
                                         gsl::span<uint32_t> indices)
    -> StaticMesh
{
  return StaticMesh{.vertex_buffer = create_vertex_buffer(device, vertices),
                    .index_buffer = create_index_buffer(device, indices),
                    .indices_size = static_cast<uint32_t>(indices.size())};
}