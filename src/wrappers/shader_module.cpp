#include "shader_module.hpp"

#include <beyond/utils/bit_cast.hpp>
#include <beyond/utils/panic.hpp>

#include <cstddef>
#include <fstream>

namespace {

[[nodiscard]] auto read_file(const std::string_view filename)
    -> std::vector<char>
{

  std::ifstream file(filename.data(), std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    // TODO(lesley): error handling
    beyond::panic("Failed to open file " + std::string(filename));
  }

  size_t file_size = static_cast<size_t>(file.tellg());
  std::vector<char> buffer;
  buffer.resize(file_size);

  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(file_size));

  return buffer;
}

} // anonymous namespace

[[nodiscard]] auto create_shader_module(VkDevice device,
                                        const std::string_view filename)
    -> VkShaderModule
{
  const auto buffer = read_file(filename);

  const VkShaderModuleCreateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .codeSize = buffer.size(),
      .pCode = beyond::bit_cast<const uint32_t*>(buffer.data()),
  };

  VkShaderModule module = nullptr;
  // TODO(lesley): error handling
  if (vkCreateShaderModule(device, &create_info, nullptr, &module) !=
      VK_SUCCESS) {
    beyond::panic("Cannot load shader\n");
  }

  return module;
}

[[nodiscard]] auto create_unique_shader_module(VkDevice device,
                                               std::string_view filename)
    -> UniqueShaderModule
{
  return UniqueShaderModule{device, create_shader_module(device, filename),
                            nullptr};
}
