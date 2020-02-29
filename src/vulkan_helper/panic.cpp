#include <cstdio>
#include <cstdlib>

#include "panic.hpp"

namespace vkh {

[[noreturn]] auto panic(std::string_view msg) noexcept -> void
{
  std::fprintf(stderr, "Panic in Vulkan Helper: %s\n", msg.data());
  std::fflush(stderr);
  std::abort();
}

} // namespace vkh
