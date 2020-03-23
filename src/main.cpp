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

  // create_frame_buffers();

  while (!window.should_close()) {
    window.poll_events();
    window.swap_buffers();
  }

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
