#include <fmt/format.h>

#include <volk.h>

#include <set>

#include "beyond/core/utils/panic.hpp"

#include "vulkan_helper/instance.hpp"
#include "vulkan_helper/panic.hpp"
#include "vulkan_helper/queue_indices.hpp"
#include "vulkan_helper/shader_module.hpp"
#include "vulkan_helper/swapchain.hpp"
#include "vulkan_helper/utils.hpp"

#include "window.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#define VKH_CHECK(call)                                                        \
  do {                                                                         \
    [[maybe_unused]] VkResult result = call;                                   \
    if (result != VK_SUCCESS) {                                                \
      ::vkh::panic(fmt::format("[{}:{}] Vulkan Fail at in {}\n", __FILE__,     \
                               __LINE__, __func__));                           \
    }                                                                          \
  } while (0)

namespace {

constexpr uint32_t max_frames_in_flight = 2;

constexpr std::array device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

[[nodiscard]] auto
check_device_extension_support(VkPhysicalDevice device) noexcept -> bool
{
  const auto available = vkh::get_vector_with<VkExtensionProperties>(
      [device](uint32_t* count, VkExtensionProperties* data) {
        vkEnumerateDeviceExtensionProperties(device, nullptr, count, data);
      });

  std::set<std::string> required(device_extensions.begin(),
                                 device_extensions.end());

  for (const auto& extension : available) {
    required.erase(static_cast<const char*>(extension.extensionName));
  }

  return required.empty();
}

// Higher is better, negative means not suitable
[[nodiscard]] auto rate_physical_device(VkPhysicalDevice device,
                                        VkSurfaceKHR surface) noexcept -> int
{
  static constexpr int failing_score = -1000;

  // If cannot find indices for all the queues, return -1000
  const auto maybe_indices = vkh::find_queue_families(device, surface);
  if (!maybe_indices) {
    return failing_score;
  }

  // If not support extension, return -1000
  if (!check_device_extension_support(device)) {
    return failing_score;
  }

  // If swapchain not adequate, return -1000
  const auto swapchain_support = vkh::query_swapchain_support(device, surface);
  if (swapchain_support.formats.empty() ||
      swapchain_support.present_modes.empty()) {
    return failing_score;
  }

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(device, &properties);

  VkPhysicalDeviceFeatures features;
  vkGetPhysicalDeviceFeatures(device, &features);

  // Biased toward discrete GPU
  int score = 0;
  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score += 100;
  }

  return score;
}

[[nodiscard]] auto pick_physical_device(VkInstance instance,
                                        VkSurfaceKHR surface) noexcept
    -> VkPhysicalDevice
{
  const auto available_devices = vkh::get_vector_with<VkPhysicalDevice>(
      [instance](uint32_t* count, VkPhysicalDevice* data) {
        return vkEnumeratePhysicalDevices(instance, count, data);
      });
  if (available_devices.empty()) {
    beyond::panic("failed to find GPUs with Vulkan support!");
  }

  using ScoredPair = std::pair<int, VkPhysicalDevice>;
  std::vector<ScoredPair> scored_pairs;
  scored_pairs.reserve(available_devices.size());
  for (const auto& device : available_devices) {
    const auto score = rate_physical_device(device, surface);
    if (score > 0) {
      scored_pairs.emplace_back(score, device);
    }
  }

  if (scored_pairs.empty()) {
    beyond::panic(
        "Vulkan failed to find GPUs with enough nessesory graphics support!");
  }

  std::sort(std::begin(scored_pairs), std::end(scored_pairs),
            [](const ScoredPair& lhs, const ScoredPair& rhs) {
              return lhs.first > rhs.first;
            });

  const auto physical_device = scored_pairs.front().second;

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(physical_device, &properties);
  std::printf("GPU: %s\n", properties.deviceName);
  std::fflush(stdout);

  // Returns the pair with highest score
  return physical_device;
}

[[nodiscard]] auto
create_logical_device(VkPhysicalDevice pd,
                      const vkh::QueueFamilyIndices& indices) noexcept
    -> VkDevice
{
  const auto unique_indices = indices.to_set();

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  queue_create_infos.resize(unique_indices.size());

  float queue_priority = 1.0f;
  std::transform(std::begin(unique_indices), std::end(unique_indices),
                 std::begin(queue_create_infos), [&](uint32_t index) {
                   return VkDeviceQueueCreateInfo{
                       .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                       .pNext = nullptr,
                       .flags = 0,
                       .queueFamilyIndex = index,
                       .queueCount = 1,
                       .pQueuePriorities = &queue_priority,
                   };
                 });

  const VkPhysicalDeviceFeatures features = {};

  const VkDeviceCreateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queueCreateInfoCount = vkh::to_u32(queue_create_infos.size()),
      .pQueueCreateInfos = queue_create_infos.data(),
#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
      .enabledLayerCount = vkh::to_u32(vkh::validation_layers.size()),
      .ppEnabledLayerNames = vkh::validation_layers.data(),
#else
      .enabledLayerCount = 0,
      .ppEnabledLayerNames = nullptr,
#endif
      .enabledExtensionCount = vkh::to_u32(device_extensions.size()),
      .ppEnabledExtensionNames = device_extensions.data(),
      .pEnabledFeatures = &features,
  };

  VkDevice device = nullptr;
  if (vkCreateDevice(pd, &create_info, nullptr, &device) != VK_SUCCESS) {
    vkh::panic("Vulkan: failed to create logical device!");
  }

  return device;
}

struct PipelineInfo {
  VkPipelineLayout pipeline_layout = nullptr;
  VkPipeline pipeline = nullptr;
};

[[nodiscard]] auto create_render_pass(VkDevice device,
                                      const vkh::Swapchain& swapchain)
    -> VkRenderPass
{
  const VkAttachmentDescription color_attachment = {
      .flags = 0,
      .format = swapchain.image_format(),
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
  };

  const VkAttachmentReference color_attachment_ref = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

  const VkSubpassDescription subpass = {
      .flags = 0,
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .inputAttachmentCount = 0,
      .pInputAttachments = nullptr,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_attachment_ref,
      .pResolveAttachments = nullptr,
      .pDepthStencilAttachment = nullptr,
      .preserveAttachmentCount = 0,
      .pPreserveAttachments = nullptr,
  };

  const VkRenderPassCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .attachmentCount = 1,
      .pAttachments = &color_attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 0,
      .pDependencies = nullptr,
  };

  VkRenderPass render_pass;
  VKH_CHECK(vkCreateRenderPass(device, &create_info, nullptr, &render_pass));
  return render_pass;
} // namespace

[[nodiscard]] auto create_graphics_pipeline(VkDevice device,
                                            const vkh::Swapchain& swapchain,
                                            VkRenderPass renderPass)
{
  const auto vert_shader =
      vkh::create_unique_shader_module("shaders/shader.vert.spv", device);
  const auto frag_shader =
      vkh::create_unique_shader_module("shaders/shader.frag.spv", device);

  const VkPipelineShaderStageCreateInfo vert_shader_stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vert_shader.get(),
      .pName = "main",
      .pSpecializationInfo = nullptr};

  const VkPipelineShaderStageCreateInfo frag_shader_stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = frag_shader.get(),
      .pName = "main",
      .pSpecializationInfo = nullptr,
  };

  const std::array shader_stages = {vert_shader_stage_info,
                                    frag_shader_stage_info};

  const VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .vertexBindingDescriptionCount = 0,
      .pVertexBindingDescriptions = nullptr,
      .vertexAttributeDescriptionCount = 0,
      .pVertexAttributeDescriptions = nullptr,
  };

  const VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE,
  };

  const VkViewport viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<float>(swapchain.extent().width),
      .height = static_cast<float>(swapchain.extent().height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };

  const VkRect2D scissor = {.offset = {0, 0}, .extent = swapchain.extent()};

  const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
  };

  const VkPipelineRasterizationStateCreateInfo rasterizer = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_CLOCKWISE,

      .depthBiasEnable = VK_FALSE,
      .depthBiasConstantFactor = 0.0f,
      .depthBiasClamp = 0.0f,
      .depthBiasSlopeFactor = 0.0f,

      .lineWidth = 1.0f,
  };

  const VkPipelineMultisampleStateCreateInfo multisampling = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = VK_FALSE,
      .minSampleShading = 0,
      .pSampleMask = nullptr,
      .alphaToCoverageEnable = VK_FALSE,
      .alphaToOneEnable = VK_FALSE,
  };

  [[maybe_unused]] VkPipelineColorBlendAttachmentState colorBlendAttachment =
      {};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  const VkPipelineColorBlendStateCreateInfo colorBlending = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_COPY,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment,
      .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
  };

  const VkPipelineLayoutCreateInfo pipeline_layout_create_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = 0,
      .pSetLayouts = nullptr,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr,
  };

  VkPipelineLayout pipeline_layout;
  VKH_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_create_info,
                                   nullptr, &pipeline_layout));

  const VkGraphicsPipelineCreateInfo pipeline_create_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stageCount = vkh::to_u32(shader_stages.size()),
      .pStages = shader_stages.data(),
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pColorBlendState = &colorBlending,
      .layout = pipeline_layout,
      .renderPass = renderPass,
      .subpass = 0,
      .basePipelineHandle = VK_NULL_HANDLE,
  };

  VkPipeline pipeline;
  VKH_CHECK(vkCreateGraphicsPipelines(
      device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &pipeline));

  // Do something
  return PipelineInfo{pipeline_layout, pipeline};
}

[[nodiscard]] auto create_frame_buffers(const VkDevice device,
                                        const vkh::Swapchain& swapchain,
                                        const VkRenderPass render_pass)
{
  std::vector<VkFramebuffer> swapchain_frame_buffers;
  swapchain_frame_buffers.resize(swapchain.image_views().size());

  const auto& swapchain_image_views = swapchain.image_views();

  VkFramebufferCreateInfo framebufferInfo = {};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = render_pass;
  framebufferInfo.attachmentCount = 1;
  framebufferInfo.width = swapchain.extent().width;
  framebufferInfo.height = swapchain.extent().height;
  framebufferInfo.layers = 1;

  for (size_t i = 0; i < swapchain_image_views.size(); i++) {
    VkImageView attachments[] = {swapchain_image_views[i]};
    framebufferInfo.pAttachments = attachments;

    VKH_CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr,
                                  &swapchain_frame_buffers[i]));
  }
  return swapchain_frame_buffers;
}

[[nodiscard]] auto create_command_pool(
    const VkDevice device,
    const vkh::QueueFamilyIndices& queue_family_indicies) noexcept
    -> VkCommandPool
{
  const VkCommandPoolCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queueFamilyIndex = queue_family_indicies.graphics_family,
  };

  VkCommandPool command_pool;
  VKH_CHECK(vkCreateCommandPool(device, &create_info, nullptr, &command_pool));
  return command_pool;
}

[[nodiscard]] auto
create_command_buffers(const VkDevice device, const VkCommandPool command_pool,
                       const VkRenderPass render_pass,
                       const VkPipeline pipeline,
                       const vkh::Swapchain& swapchain,
                       const std::vector<VkFramebuffer>& framebuffers) noexcept
    -> std::vector<VkCommandBuffer>
{
  std::vector<VkCommandBuffer> command_buffers;
  command_buffers.resize(framebuffers.size());

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = command_pool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = vkh::to_u32(command_buffers.size());

  VKH_CHECK(
      vkAllocateCommandBuffers(device, &allocInfo, command_buffers.data()));

  for (size_t i = 0; i < command_buffers.size(); i++) {
    auto& command_buffer = command_buffers[i];

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VKH_CHECK(vkBeginCommandBuffer(command_buffer, &beginInfo));

    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = render_pass;
    renderPassInfo.framebuffer = framebuffers[i];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapchain.extent();

    VkClearValue clear_color;
    clear_color.color = {0.0f, 0.0f, 0.0f, 1.0f};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clear_color;

    vkCmdBeginRenderPass(command_buffer, &renderPassInfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline);

    vkCmdDraw(command_buffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(command_buffer);

    VKH_CHECK(vkEndCommandBuffer(command_buffer));
  }

  return command_buffers;
}

void draw_frame(VkDevice device, const vkh::Swapchain& swapchain,
                VkQueue graphics_queue, VkQueue present_queue,
                const std::vector<VkFence>& in_flight_fences,
                std::vector<VkFence> images_in_flight_fences,
                const std::vector<VkSemaphore>& image_available_semaphores,
                const std::vector<VkSemaphore>& render_finished_semaphores,
                const std::vector<VkCommandBuffer>& command_buffers) noexcept
{
  static uint32_t current_frame = 0;

  vkWaitForFences(device, 1, &in_flight_fences[current_frame], VK_TRUE,
                  UINT64_MAX);

  uint32_t image_index;
  vkAcquireNextImageKHR(device, swapchain.get(), UINT64_MAX,
                        image_available_semaphores[current_frame],
                        VK_NULL_HANDLE, &image_index);

  if (images_in_flight_fences[image_index] != VK_NULL_HANDLE) {
    vkWaitForFences(device, 1, &images_in_flight_fences[image_index], VK_TRUE,
                    UINT64_MAX);
  }
  images_in_flight_fences[image_index] = in_flight_fences[current_frame];

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore waitSemaphores[] = {image_available_semaphores[current_frame]};
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffers[image_index];

  VkSemaphore signalSemaphores[] = {render_finished_semaphores[current_frame]};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  vkResetFences(device, 1, &in_flight_fences[current_frame]);

  VKH_CHECK(vkQueueSubmit(graphics_queue, 1, &submitInfo,
                          in_flight_fences[current_frame]));

  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = signalSemaphores;

  VkSwapchainKHR swapChains[] = {swapchain.get()};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;

  presentInfo.pImageIndices = &image_index;

  vkQueuePresentKHR(present_queue, &presentInfo);

  current_frame = (current_frame + 1) % max_frames_in_flight;
}

} // anonymous namespace

auto main() -> int
{
  if (volkInitialize() != VK_SUCCESS) {
    beyond::panic("Cannot find a Vulkan Loader in the system!");
  }

  Window window(1024, 768, "Vulkan Renderer");

  auto instance = vkh::create_instance(
      window.title().c_str(), window.get_required_instance_extensions());

  VkSurfaceKHR surface;
  window.create_vulkan_surface(instance, nullptr, surface);

#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
  auto debug_messenger = vkh::create_debug_messenger(instance);
#endif

  const auto physical_device = pick_physical_device(instance, surface);
  const auto queue_family_indices = [&]() {
    auto indices = vkh::find_queue_families(physical_device, surface);
    if (!indices) {
      vkh::panic("Cannot find a physical device that satisfy all the queue "
                 "family indices requirements");
    }
    return *indices;
  }();

  auto device = create_logical_device(physical_device, queue_family_indices);
  volkLoadDevice(device);

  const auto get_device_queue = [device](std::uint32_t family_index,
                                         std::uint32_t index) {
    VkQueue queue;
    vkGetDeviceQueue(device, family_index, index, &queue);
    return queue;
  };
  [[maybe_unused]] auto graphics_queue =
      get_device_queue(queue_family_indices.graphics_family, 0);
  [[maybe_unused]] auto present_queue =
      get_device_queue(queue_family_indices.present_family, 0);
  [[maybe_unused]] auto compute_queue =
      get_device_queue(queue_family_indices.compute_family, 0);

  VmaAllocatorCreateInfo allocator_info{};
  allocator_info.physicalDevice = physical_device;
  allocator_info.device = device;
  VmaAllocator allocator;

  VKH_CHECK(vmaCreateAllocator(&allocator_info, &allocator));

  vkh::Swapchain swapchain(physical_device, device, surface,
                           queue_family_indices);

  auto render_pass = create_render_pass(device, swapchain);
  auto [pipeline_layout, pipeline] =
      create_graphics_pipeline(device, swapchain, render_pass);
  auto framebuffers = create_frame_buffers(device, swapchain, render_pass);

  auto command_pool = create_command_pool(device, queue_family_indices);
  auto command_buffers = create_command_buffers(
      device, command_pool, render_pass, pipeline, swapchain, framebuffers);

  std::vector<VkSemaphore> image_available_semaphores(max_frames_in_flight);
  std::vector<VkSemaphore> render_finished_semaphores(max_frames_in_flight);
  std::vector<VkFence> in_flight_fence(max_frames_in_flight);
  std::vector<VkFence> image_in_flight_fence(swapchain.image_views().size(),
                                             VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_create_info = {};
  semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < max_frames_in_flight; i++) {
    VKH_CHECK(vkCreateSemaphore(device, &semaphore_create_info, nullptr,
                                &image_available_semaphores[i]));
    VKH_CHECK(vkCreateSemaphore(device, &semaphore_create_info, nullptr,
                                &render_finished_semaphores[i]));
    VKH_CHECK(vkCreateFence(device, &fence_info, nullptr, &in_flight_fence[i]));
  }

  while (!window.should_close()) {
    draw_frame(device, swapchain, graphics_queue, present_queue,
               in_flight_fence, image_in_flight_fence,
               image_available_semaphores, render_finished_semaphores,
               command_buffers);

    window.poll_events();
    window.swap_buffers();
  }

  vkDeviceWaitIdle(device);

  for (size_t i = 0; i < max_frames_in_flight; i++) {
    vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
    vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
    vkDestroyFence(device, in_flight_fence[i], nullptr);
  }

  vkDestroyCommandPool(device, command_pool, nullptr);

  for (auto framebuffer : framebuffers) {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }

  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyRenderPass(device, render_pass, nullptr);

  swapchain.reset();

  vmaDestroyAllocator(allocator);
  vkDestroyDevice(device, nullptr);
#ifdef VULKAN_HELPER_ENABLE_VALIDATION_LAYER
  vkDestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
#endif
  vkDestroySurfaceKHR(instance, surface, nullptr);
  vkDestroyInstance(instance, nullptr);
}
