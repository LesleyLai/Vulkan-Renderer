#ifndef VULKAN_HELPER_UNIQUE_RESOURCE_HPP
#define VULKAN_HELPER_UNIQUE_RESOURCE_HPP

#include <vulkan/vulkan_core.h>

#include <utility>

namespace vkh {

template <typename T,
          void (*deleter)(VkDevice, T, const VkAllocationCallbacks*)>
class UniqueResource {
public:
  UniqueResource() noexcept = default;
  UniqueResource(VkDevice device, T resource,
                 const VkAllocationCallbacks* allocator_ptr = nullptr) noexcept
      : device_{device}, allocator_ptr_{allocator_ptr}, resource_{
                                                            std::move(resource)}
  {
  }

  ~UniqueResource() noexcept
  {
    delete_without_reset();
  }

  UniqueResource(const UniqueResource& other) noexcept = delete;
  auto operator=(const UniqueResource& other) & noexcept
      -> UniqueResource& = delete;

  UniqueResource(UniqueResource&& other) noexcept
      : device_{std::exchange(other.device_, nullptr)},
        allocator_ptr_{std::exchange(other.allocator_ptr_, nullptr)},
        resource_{std::exchange(other.resource_, nullptr)}
  {
  }

  auto operator=(UniqueResource&& other) & noexcept -> UniqueResource&
  {
    if (resource_ != other.resource_) {
      delete_without_reset();
      device_ = std::exchange(other.device_, nullptr);
      allocator_ptr_ = std::exchange(other.allocator_ptr_, nullptr);
      resource_ = std::exchange(other.resource_, nullptr);
    }

    return *this;
  }

  auto reset() noexcept -> void
  {
    delete_without_reset();
    device_ = nullptr;
    allocator_ptr_ = nullptr;
    resource_ = nullptr;
  }

  auto get() const noexcept -> T
  {
    return resource_;
  }

  explicit operator T() const noexcept
  {
    return resource_;
  }

  auto swap(UniqueResource& rhs) noexcept -> void
  {
    std::swap(device_, rhs.device_);
    std::swap(allocator_ptr_, rhs.allocator_ptr_);
    std::swap(resource_, rhs.resource_);
  }

  friend auto swap(UniqueResource& lhs, UniqueResource& rhs) noexcept -> void
  {
    lhs.swap(rhs);
  }

private:
  auto delete_without_reset() noexcept -> void
  {
    if (resource_ != nullptr) {
      deleter(device_, resource_, allocator_ptr_);
    }
  }

  VkDevice device_ = nullptr;
  const VkAllocationCallbacks* allocator_ptr_ = nullptr;
  T resource_;
};

} // namespace vkh

#endif // VULKAN_HELPER_UNIQUE_RESOURCE_HPP
