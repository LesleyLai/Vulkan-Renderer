#pragma once

#ifndef BEYOND_VULKAN_HELPER_UTILS_HPP
#define BEYOND_VULKAN_HELPER_UTILS_HPP

#include <cstdint>
#include <vector>

namespace beyond::vkh {

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

/// @brief Casts a number into `std::uint32_t`
template <typename T> constexpr auto to_u32(T value) noexcept -> std::uint32_t
{
  return static_cast<std::uint32_t>(value);
}

} // namespace beyond::vkh

#endif // BEYOND_VULKAN_HELPER_UTILS_HPP
