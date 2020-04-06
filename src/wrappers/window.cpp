#include <algorithm>
#include <utility>

// Supress windows Macro redefinition of GLFW
#ifdef WIN32
#include <windows.h>
#endif

#include <beyond/core/utils/panic.hpp>

#include "window.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

Window::~Window() noexcept
{
  if (pimpl_) {
    glfwTerminate();
  }
}

Window::Window(Window&& other) noexcept = default;
auto Window::operator=(Window&& other) noexcept -> Window& = default;

struct WindowImpl {
  GLFWwindow* data_ = nullptr;

  explicit WindowImpl(GLFWwindow* data) : data_{data} {}
  ~WindowImpl()
  {
    glfwDestroyWindow(data_);
  }
};

Window::Window(int width, int height, std::string title) noexcept
    : title_{std::move(title)}
{
  if (glfwInit() != GLFW_TRUE) {
    beyond::panic("Cannot initialize the GLFW platform!");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  auto glfw_window =
      glfwCreateWindow(width, height, title_.data(), nullptr, nullptr);

  if (!glfw_window) {
    beyond::panic("Cannot Create a GLFW Window");
  }

  pimpl_ = std::make_unique<WindowImpl>(glfw_window);
  glfwMakeContextCurrent(pimpl_->data_);
}

auto Window::poll_events() noexcept -> void
{
  glfwPollEvents();
}

auto Window::swap_buffers() noexcept -> void
{
  glfwSwapBuffers(pimpl_->data_);
}

[[nodiscard]] auto Window::should_close() const noexcept -> bool
{
  return static_cast<bool>(glfwWindowShouldClose(pimpl_->data_));
}

[[nodiscard]] auto Window::get() const noexcept -> GLFWwindow*
{
  return pimpl_->data_;
}

/// @brief Get the extensions needed for the vulkan instance
[[nodiscard]] auto Window::get_required_instance_extensions() noexcept
    -> std::vector<const char*>
{
  uint32_t glfw_extension_count = 0;
  const char** glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

  std::vector<const char*> extensions;
  std::copy_n(glfw_extensions, glfw_extension_count,
              std::back_inserter(extensions));
  return extensions;
}

auto Window::create_vulkan_surface(
    VkInstance instance, const VkAllocationCallbacks* allocator) noexcept
    -> VkSurfaceKHR
{
  VkSurfaceKHR surface;
  if (glfwCreateWindowSurface(instance, pimpl_->data_, allocator, &surface) !=
      VK_SUCCESS) {
    beyond::panic("Failed to create Vulkan window surface!");
  }
  return surface;
}