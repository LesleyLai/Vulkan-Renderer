#pragma once

#ifndef VULKAN_HELPER_PANIC_HPP
#define VULKAN_HELPER_PANIC_HPP

#include <string_view>

namespace vkh {

[[noreturn]] auto panic(std::string_view msg) noexcept -> void;

}

#endif // VULKAN_HELPER_PANIC_HPP
