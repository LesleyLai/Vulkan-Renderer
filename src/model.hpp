#ifndef VULKAN_RENDERER_MODEL_HPP
#define VULKAN_RENDERER_MODEL_HPP

#include <string_view>
#include <vector>

#include "mesh.hpp"

class Model {
  std::vector<StaticMesh> meshes_;

  explicit Model(std::vector<StaticMesh>&& meshes) : meshes_{std::move(meshes)}
  {
  }

public:
  Model() = default;

  [[nodiscard]] auto meshes() -> const std::vector<StaticMesh>&
  {
    return meshes_;
  }

  [[nodiscard]] static auto load(vkh::GPUDevice& device, const char* path)
      -> Model;
};

#endif // VULKAN_RENDERER_MODEL_HPP
