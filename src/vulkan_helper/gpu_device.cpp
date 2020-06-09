#include "gpu_device.hpp"
#include "../window.hpp"

#include <array>
#include <cstring>
#include <iostream>

#include <fmt/format.h>

#include "../utils/utils.hpp"

#include <beyond/utils/panic.hpp>

namespace {

auto get_max_usable_sample_count(const vkb::PhysicalDevice& pd) noexcept
    -> VkSampleCountFlagBits
{
  VkSampleCountFlags counts =
      pd.properties.limits.framebufferColorSampleCounts &
      pd.properties.limits.framebufferDepthSampleCounts;
  if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
  if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
  if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
  if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
  if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
  if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

  return VK_SAMPLE_COUNT_1_BIT;
}

} // namespace

namespace vkh {

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
  VkPhysicalDeviceFeatures features = {};
  features.samplerAnisotropy = VK_TRUE;
  selector.set_surface(surface_)
      .set_minimum_version(1, 1)
      .set_required_features(features);
  auto phys_ret = selector.select();
  if (!phys_ret) {
    beyond::panic(
        fmt::format("Error: {}", vkb::to_string(phys_ret.error().type)));
  }

  physical_device_ = phys_ret->physical_device;
  msaa_sample_count_ = get_max_usable_sample_count(*phys_ret);

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

  queue_family_indices_ = {
      .graphics_family =
          vkb_device.get_queue_index(vkb::QueueType::graphics).value(),
      .present_family =
          vkb_device.get_queue_index(vkb::QueueType::present).value(),
      .compute_family =
          vkb_device.get_queue_index(vkb::QueueType::compute).value()};

  const VmaAllocatorCreateInfo allocator_info{
      .physicalDevice = physical_device_,
      .device = device_,
  };

  if (vmaCreateAllocator(&allocator_info, &allocator_) != VK_SUCCESS) {
    beyond::panic("Cannot create vma allocator");
  }

  const VkCommandPoolCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family_indices_.graphics_family,
  };

  if (vkCreateCommandPool(device_, &create_info, nullptr,
                          &graphics_command_pool_) != VK_SUCCESS) {
    beyond::panic("Cannot create graphics command pool");
  }
}

GPUDevice::~GPUDevice() noexcept
{
  vkDestroyCommandPool(device_, graphics_command_pool_, nullptr);

  vmaDestroyAllocator(allocator_);
  vkDestroyDevice(device_, nullptr);

  vkDestroySurfaceKHR(instance_.instance, surface_, nullptr);
  vkb::destroy_instance(instance_);
}

auto GPUDevice::wait_idle() const noexcept -> void
{
  vkDeviceWaitIdle(this->device());
}

} // namespace vkh