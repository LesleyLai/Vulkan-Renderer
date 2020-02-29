#pragma once

#ifndef VULKAN_HELPER_INSTANCE_HPP
#define VULKAN_HELPER_INSTANCE_HPP

#include <volk.h>

#include <vector>

namespace vkh {

[[nodiscard]] auto
create_instance(const char* title,
                std::vector<const char*> required_extensions) noexcept
    -> VkInstance;

}

#endif // VULKAN_HELPER_INSTANCE_HPP
