#pragma once

#ifndef VULKAN_HELPER_UTILS_HPP
#define VULKAN_HELPER_UTILS_HPP

#include <cstdint>
#include <vector>

namespace vkh {

/// @brief Transforms the two stage query vulkan function into directly return
/// vector
template <typename T, typename F> auto get_vector_with(F func) -> std::vector<T>
{
  std::uint32_t count;
  func(&count, nullptr);

  std::vector<T> vec(count);
  func(&count, vec.data());

  return vec;
}

/// @brief Casts a number into `uint32_t`
template <typename T> constexpr auto to_u32(T value) noexcept -> std::uint32_t
{
  return static_cast<std::uint32_t>(value);
}

} // namespace vkh

#endif // VULKAN_HELPER_UTILS_HPP
