#include "gpu_device.hpp"
#include "window.hpp"

#include <array>
#include <cstring>
#include <iostream>

#include <fmt/format.h>

#include "../vulkan_helper/panic.hpp"
#include "../vulkan_helper/utils.hpp"

#include <beyond/core/utils/panic.hpp>

GPUDevice::GPUDevice(Window& window,
                     ValidationLayerSetting validation_layer_setting) noexcept
{
  vkb::InstanceBuilder instance_builder;
  instance_builder.require_api_version(1, 2, 0).set_app_name(
      "Lesley Vulkan Renderer");

  if (validation_layer_setting == ValidationLayerSetting::enable) {
    instance_builder.request_validation_layers().use_default_debug_messenger();
  }

  auto instance_ret = instance_builder.build();
  if (!instance_ret) {
    beyond::panic(
        fmt::format("Failed to create a vulkan instance with error message: {}",
                    vkb::to_string(instance_ret.error().type)));
  }

  instance_ = *instance_ret;

  surface_ = window.create_vulkan_surface(instance_.instance, nullptr);

  vkb::PhysicalDeviceSelector selector{*instance_ret};
  selector.set_surface(surface_).set_minimum_version(1, 1);
  auto phys_ret = selector.select();
  if (!phys_ret) {
    beyond::panic(
        fmt::format("Error: {}", vkb::to_string(phys_ret.error().type)));
  }

  physical_device_ = phys_ret->physical_device;

  vkb::DeviceBuilder device_builder{phys_ret.value()};
  auto dev_ret = device_builder.build();
  if (!dev_ret) {
    beyond::panic(
        fmt::format("Error: {}", vkb::to_string(dev_ret.error().type)));
  }
  vkb::Device vkb_device = dev_ret.value();

  device_ = vkb_device.device;

  auto graphics_queue_ret = vkb_device.get_queue(vkb::QueueType::graphics);
  if (!graphics_queue_ret.has_value()) {
    beyond::panic(fmt::format("Error: {}",
                              vkb::to_string(graphics_queue_ret.error().type)));
  }
  graphics_queue_ = graphics_queue_ret.value();

  auto present_queue_ret = vkb_device.get_queue(vkb::QueueType::present);
  if (!present_queue_ret.has_value()) {
    beyond::panic(fmt::format("Error: {}",
                              vkb::to_string(present_queue_ret.error().type)));
  }
  present_queue_ = present_queue_ret.value();

  auto compute_queue_ret = vkb_device.get_queue(vkb::QueueType::graphics);
  if (!graphics_queue_ret.has_value()) {
    beyond::panic(fmt::format("Error: {}",
                              vkb::to_string(graphics_queue_ret.error().type)));
  }
  compute_queue_ = graphics_queue_ret.value();

  VmaAllocatorCreateInfo allocator_info{};
  allocator_info.physicalDevice = physical_device_;
  allocator_info.device = device_;

  if (vmaCreateAllocator(&allocator_info, &allocator_) != VK_SUCCESS) {
    beyond::panic("Big bad");
  }
}

GPUDevice::~GPUDevice() noexcept
{
  vmaDestroyAllocator(allocator_);
  vkDestroyDevice(device_, nullptr);

  vkDestroySurfaceKHR(instance_.instance, surface_, nullptr);
  vkb::destroy_instance(instance_);
}

auto GPUDevice::wait_idle() const noexcept -> void
{
  vkDeviceWaitIdle(this->vk_device());
}