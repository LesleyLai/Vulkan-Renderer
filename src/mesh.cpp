#include "mesh.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace std {
template <> struct hash<Vertex> {
  [[nodiscard]] auto operator()(Vertex const& vertex) const noexcept -> size_t
  {
    return ((hash<glm::vec3>()(vertex.pos) ^
             (hash<glm::vec3>()(vertex.color) << 1u)) >>
            1u) ^
           (hash<glm::vec2>()(vertex.tex_coord) << 1u);
  }
};
} // namespace std

[[nodiscard]] auto load_mesh(const char* path) -> Mesh
{
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path)) {
    throw std::runtime_error(err);
  }

  std::unordered_map<Vertex, uint32_t> unique_vertices{};

  Mesh mesh;

  for (const auto& shape : shapes) {
    for (const auto& index : shape.mesh.indices) {
      Vertex vertex = {
          .pos =
              {attrib.vertices[static_cast<size_t>(3 * index.vertex_index + 0)],
               attrib.vertices[static_cast<size_t>(3 * index.vertex_index + 1)],
               attrib
                   .vertices[static_cast<size_t>(3 * index.vertex_index + 2)]},
          .color = {1.0f, 1.0f, 1.0f},
          .tex_coord = {attrib.texcoords[static_cast<size_t>(
                            2 * index.texcoord_index + 0)],
                        1.0f - attrib.texcoords[static_cast<size_t>(
                                   2 * index.texcoord_index + 1)]},
      };

      if (unique_vertices.count(vertex) == 0) {
        unique_vertices[vertex] = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(vertex);
      }

      mesh.indices.push_back(unique_vertices[vertex]);
    }
  }

  return mesh;
}