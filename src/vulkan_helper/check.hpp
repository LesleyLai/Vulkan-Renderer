#ifndef VULKAN_RENDERER_CHECK_HPP
#define VULKAN_RENDERER_CHECK_HPP

#include <beyond/utils/panic.hpp>

#include <fmt/format.h>

#define VKH_CHECK(call)                                                        \
  do {                                                                         \
    VkResult result = call;                                                    \
    if (result != VK_SUCCESS) {                                                \
      ::beyond::panic(fmt::format("[{}:{}] Vulkan Fail at in {}\n", __FILE__,  \
                                  __LINE__, __func__));                        \
    }                                                                          \
  } while (0)

#endif // VULKAN_RENDERER_CHECK_HPP
