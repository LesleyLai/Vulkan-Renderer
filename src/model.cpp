#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "model.hpp"

#include <beyond/utils/panic.hpp>

namespace {

[[nodiscard]] StaticMesh process_mesh(vkh::GPUDevice& device,
                                      VkCommandPool command_pool, VkQueue queue,
                                      const aiMesh& mesh, const aiScene& scene)
{
  std::vector<Vertex> vertices;
  vertices.reserve(mesh.mNumVertices);

  std::vector<uint32_t> indices;
  indices.reserve(mesh.mNumFaces * 3);

  // TODO: Check for null cases

  for (uint32_t i = 0; i < mesh.mNumVertices; i++) {
    vertices.push_back(
        {.pos = {mesh.mVertices[i].x, mesh.mVertices[i].y, mesh.mVertices[i].z},
         .normal = {mesh.mNormals[i].x, mesh.mNormals[i].y, mesh.mNormals[i].z},
         .tex_coord = {mesh.mTextureCoords[0][i].x,
                       mesh.mTextureCoords[0][i].y}});
  }

  for (uint32_t i = 0; i < mesh.mNumFaces; i++) {
    const aiFace face = mesh.mFaces[i];
    for (uint32_t j = 0; j < face.mNumIndices; j++)
      indices.push_back(face.mIndices[j]);
  }

  if (mesh.mMaterialIndex < scene.mNumMaterials) {
    [[maybe_unused]] aiMaterial* material =
        scene.mMaterials[mesh.mMaterialIndex];

    aiString diffuse;
    material->GetTexture(aiTextureType_DIFFUSE, 0, &diffuse);
    fmt::print("Texture path {}\n", diffuse.C_Str());

    if (const aiTexture* diffuse_texture =
            scene.GetEmbeddedTexture(diffuse.C_Str());
        diffuse_texture) {
    } else {
      beyond::panic("Cannot load diffuse texture\n");
    }

  } else {
    beyond::panic("No material!");
  }

  // TODO: Loading materials

  return create_mesh_from_data(device, command_pool, queue, vertices, indices);
}

auto process_node(vkh::GPUDevice& device, VkCommandPool command_pool,
                  VkQueue queue, const aiNode& node, const aiScene& scene,
                  std::vector<StaticMesh>& meshes) -> void
{
  fmt::print("Node: {}\n", node.mName.C_Str());

  for (unsigned int i = 0; i < node.mNumMeshes; i++) {
    [[maybe_unused]] const aiMesh& mesh = *scene.mMeshes[node.mMeshes[i]];
    fmt::print("Mesh: {}\n", mesh.mName.C_Str());

    meshes.push_back(process_mesh(device, command_pool, queue, mesh, scene));
  }

  // then do the same for each of its children
  for (uint32_t i = 0; i < node.mNumChildren; ++i) {
    process_node(device, command_pool, queue, *node.mChildren[i], scene,
                 meshes);
  }
}

} // namespace

[[nodiscard]] auto Model::load(vkh::GPUDevice& device,
                               VkCommandPool command_pool, VkQueue queue,
                               const char* path) -> Model
{
  Assimp::Importer importer;
  const aiScene* scene =
      importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
      !scene->mRootNode) {
    beyond::panic("Error, cannot read scene from {}", path);
  }
  // directory = path.substr(0, path.find_last_of('/'));

  std::vector<StaticMesh> meshes;
  process_node(device, command_pool, queue, *scene->mRootNode, *scene, meshes);

  return Model{std::move(meshes)};
}
