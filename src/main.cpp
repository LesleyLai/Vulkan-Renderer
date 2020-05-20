#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "mesh.hpp"

#include "vulkan_helper/buffer.hpp"
#include "vulkan_helper/check.hpp"
#include "vulkan_helper/gpu_device.hpp"
#include "vulkan_helper/shader_module.hpp"
#include "vulkan_helper/single_time_command.hpp"
#include "vulkan_helper/swapchain.hpp"

#include "vulkan_helper/image.hpp"

#include "window.hpp"

#include <beyond/math/angle.hpp>
#include <beyond/utils/bit_cast.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

constexpr int init_width = 800;
constexpr int init_height = 600;

constexpr const char* model_path = "models/chalet.obj";

constexpr int max_frames_in_flight = 2;

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

auto generate_uv_sphere(vkh::GPUDevice& device, VkCommandPool command_pool,
                        VkQueue queue) -> StaticMesh
{
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  constexpr uint32_t x_segments_count = 32;
  constexpr uint32_t y_segments_count = 32;
  constexpr auto pi = glm::pi<float>();
  for (unsigned int y = 0; y <= y_segments_count; ++y) {
    const auto y_segment =
        static_cast<float>(y) / static_cast<float>(y_segments_count);
    const auto theta = y_segment * pi;

    for (unsigned int x = 0; x <= x_segments_count; ++x) {
      const auto x_segment =
          static_cast<float>(x) / static_cast<float>(x_segments_count);

      const auto phi = x_segment * 2.0f * pi;

      const float x_pos = std::cos(phi) * std::sin(theta);
      const float y_pos = std::sin(phi) * std::sin(theta);
      const float z_pos = std::cos(theta);

      vertices.push_back({
          .pos = glm::vec3(x_pos, y_pos, z_pos),
          .normal = glm::vec3(x_pos, y_pos, z_pos),
          .tex_coord = glm::vec2(x_segment, y_segment),
      });
    }
  }

  for (uint32_t y = 0; y < y_segments_count; ++y) {
    for (uint32_t x = 0; x < x_segments_count; ++x) {}
  }

  bool odd_row = false;
  for (unsigned int y = 0; y < x_segments_count; ++y) {
    if (!odd_row) // even rows: y == 0, y == 2; and so on
    {
      for (unsigned int x = 0; x <= y_segments_count; ++x) {
        indices.push_back(y * (y_segments_count + 1) + x);
        indices.push_back((y + 1) * (y_segments_count + 1) + x);
      }
    } else {
      for (int x = y_segments_count; x >= 0; --x) {
        const auto ux = static_cast<uint32_t>(x);
        indices.push_back((y + 1) * (y_segments_count + 1) + ux);
        indices.push_back(y * (y_segments_count + 1) + ux);
      }
    }
    odd_row = !odd_row;
  }

  return create_mesh_from_data(device, command_pool, queue, vertices, indices);
}

struct Texture {
  uint32_t mip_levels{};
  vkh::UniqueImage image{};
  VkImageView image_view{};
  VkSampler sampler{};
};

class Application {
public:
  Application() : window_{init_width, init_height, "Vulkan"}, device_{window_}
  {
    glfwSetWindowUserPointer(window_.get(), this);
    glfwSetFramebufferSizeCallback(window_.get(), framebuffer_resize_callback);
    glfwSetKeyCallback(window_.get(), key_callback);

    init_vulkan();
  }

  ~Application() noexcept
  {
    cleanup();
  }

  Application(const Application&) = delete;
  auto operator=(const Application&) = delete;
  Application(Application&&) = delete;
  auto operator=(Application&&) = delete;

  void run()
  {
    while (!window_.should_close()) {
      glfwPollEvents();
      draw_frame();
    }

    device_.wait_idle();
  }

private:
  Window window_;
  vkh::GPUDevice device_;
  vkh::Swapchain swapchain_;
  std::vector<VkFramebuffer> swapchain_framebuffers_;

  VkRenderPass render_pass_{};
  VkDescriptorSetLayout descriptor_set_layout_{};
  VkPipelineLayout pipeline_layout_{};
  VkPipeline graphics_pipeline_{};

  VkCommandPool graphics_command_pool_{};

  VkImage color_image_{};
  VmaAllocation color_image_memory_{};
  VkImageView color_image_view_{};

  VkImage depth_image_{};
  VmaAllocation depth_image_memory_{};
  VkImageView depth_image_view_{};

  Texture albedo_texture_{};
  Texture normal_texture_{};
  Texture metallic_texture_{};
  Texture roughness_texture_{};

  StaticMesh mesh_;
  std::vector<vkh::UniqueBuffer> uniform_buffers_;

  VkDescriptorPool descriptor_pool_{};
  std::vector<VkDescriptorSet> descriptor_sets_;

  std::vector<VkCommandBuffer> command_buffers_;

  std::vector<VkSemaphore> image_available_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> in_flight_fences_;
  std::vector<VkFence> images_in_flight_;
  size_t current_frame_ = 0;

  beyond::Radian rotation_x_{0.0f};
  beyond::Radian rotation_z_{0.0f};

  bool framebuffer_resized_ = false;

  static void framebuffer_resize_callback(GLFWwindow* window, int /*width*/,
                                          int /*height*/)
  {
    auto* app =
        beyond::bit_cast<Application*>(glfwGetWindowUserPointer(window));
    app->framebuffer_resized_ = true;
  }

  static void key_callback(GLFWwindow* window, int key, int /*scancode*/,
                           int action, int /*mods*/)
  {
    using namespace beyond::literals;

    auto* app =
        beyond::bit_cast<Application*>(glfwGetWindowUserPointer(window));

    switch (action) {
    case GLFW_PRESS:
      [[fallthrough]];
    case GLFW_REPEAT:
      switch (key) {
      case GLFW_KEY_W:
        app->rotation_x_ -= 0.1_rad;
        break;
      case GLFW_KEY_S:
        app->rotation_x_ += 0.1_rad;
        break;
      case GLFW_KEY_A:
        app->rotation_z_ -= 0.1_rad;
        break;
      case GLFW_KEY_D:
        app->rotation_z_ += 0.1_rad;
        break;
      default:
        break;
      }
      break;
    default:
      break;
    }
  }

  void init_vulkan()
  {
    swapchain_ = vkh::Swapchain(device_);
    create_render_pass();
    create_descriptor_set_layout();
    create_graphics_pipeline();
    create_command_pool();
    create_color_resources();
    create_depth_resources();
    create_framebuffers();
    albedo_texture_ = create_texture_image(
        "textures/rustediron1-alt2-bl/rustediron2_basecolor.png");
    normal_texture_ = create_texture_image(
        "textures/rustediron1-alt2-bl/rustediron2_normal.png");
    metallic_texture_ = create_texture_image(
        "textures/rustediron1-alt2-bl/rustediron2_metallic.png");
    roughness_texture_ = create_texture_image(
        "textures/rustediron1-alt2-bl/rustediron2_roughness.png");

    mesh_ = generate_uv_sphere(device_, graphics_command_pool_,
                               device_.graphics_queue());
    create_uniform_buffers();
    create_descriptor_pool();
    create_descriptor_sets();
    create_command_buffers();
    create_sync_objects();
  }

  void cleanup_swapchain()
  {
    vkDestroyImageView(device_.device(), depth_image_view_, nullptr);
    vmaDestroyImage(device_.allocator(), depth_image_, depth_image_memory_);

    vkDestroyImageView(device_.device(), color_image_view_, nullptr);
    vmaDestroyImage(device_.allocator(), color_image_, color_image_memory_);

    for (auto* framebuffer : swapchain_framebuffers_) {
      vkDestroyFramebuffer(device_.device(), framebuffer, nullptr);
    }

    swapchain_.reset();

    vkFreeCommandBuffers(device_.device(), graphics_command_pool_,
                         static_cast<uint32_t>(command_buffers_.size()),
                         command_buffers_.data());

    vkDestroyPipeline(device_.device(), graphics_pipeline_, nullptr);
    vkDestroyPipelineLayout(device_.device(), pipeline_layout_, nullptr);
    vkDestroyRenderPass(device_.device(), render_pass_, nullptr);

    for (size_t i = 0; i < swapchain_.images().size(); i++) {
      uniform_buffers_[i].reset();
    }

    vkDestroyDescriptorPool(device_.device(), descriptor_pool_, nullptr);
  }

  void cleanup() noexcept
  {
    cleanup_swapchain();

    auto destroy_texture = [device = device_.device()](Texture& texture) {
      vkDestroySampler(device, texture.sampler, nullptr);
      vkDestroyImageView(device, texture.image_view, nullptr);
      texture.image.reset();
    };

    destroy_texture(roughness_texture_);
    destroy_texture(metallic_texture_);
    destroy_texture(normal_texture_);
    destroy_texture(albedo_texture_);

    vkDestroyDescriptorSetLayout(device_.device(), descriptor_set_layout_,
                                 nullptr);

    mesh_.index_buffer.reset();
    mesh_.vertex_buffer.reset();

    for (size_t i = 0; i < max_frames_in_flight; i++) {
      vkDestroySemaphore(device_.device(), render_finished_semaphores_[i],
                         nullptr);
      vkDestroySemaphore(device_.device(), image_available_semaphores_[i],
                         nullptr);
      vkDestroyFence(device_.device(), in_flight_fences_[i], nullptr);
    }

    vkDestroyCommandPool(device_.device(), graphics_command_pool_, nullptr);
  }

  void recreate_swapchain()
  {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window_.get(), &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window_.get(), &width, &height);
      glfwWaitEvents();
    }

    vkDeviceWaitIdle(device_.device());

    cleanup_swapchain();

    swapchain_ = vkh::Swapchain(device_);

    create_render_pass();
    create_graphics_pipeline();
    create_color_resources();
    create_depth_resources();
    create_framebuffers();
    create_uniform_buffers();
    create_descriptor_pool();
    create_descriptor_sets();
    create_command_buffers();
  }

  void create_render_pass()
  {
    const VkAttachmentDescription color_attachment_desc = {
        .format = swapchain_.image_format(),
        .samples = device_.msaa_sample_count(),
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    const VkAttachmentDescription depth_attachment_desc = {
        .format = find_depth_format(),
        .samples = device_.msaa_sample_count(),
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    const VkAttachmentDescription color_attachment_resolve_desc = {
        .format = swapchain_.image_format(),
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    const VkAttachmentReference color_attachment_ref = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    const VkAttachmentReference depth_attachment_ref = {
        .attachment = 1,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    const VkAttachmentReference color_attachment_resolve_ref = {
        .attachment = 2,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    const VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
        .pResolveAttachments = &color_attachment_resolve_ref,
        .pDepthStencilAttachment = &depth_attachment_ref,
    };

    const VkSubpassDependency dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    const std::array attachments = {color_attachment_desc,
                                    depth_attachment_desc,
                                    color_attachment_resolve_desc};
    const VkRenderPassCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    if (vkCreateRenderPass(device_.device(), &create_info, nullptr,
                           &render_pass_) != VK_SUCCESS) {
      beyond::panic("failed to create render pass!");
    }
  }

  void create_descriptor_set_layout()
  {
    const VkDescriptorSetLayoutBinding ubo_layout_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .pImmutableSamplers = nullptr,
    };

    const VkDescriptorSetLayoutBinding albedo_sampler_layout_binding = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = nullptr,
    };

    const VkDescriptorSetLayoutBinding normal_sampler_layout_binding = {
        .binding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = nullptr,
    };

    const VkDescriptorSetLayoutBinding metallic_sampler_layout_binding = {
        .binding = 3,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = nullptr,
    };

    const VkDescriptorSetLayoutBinding roughness_sampler_layout_binding = {
        .binding = 4,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = nullptr,
    };

    const std::array bindings = {
        ubo_layout_binding, albedo_sampler_layout_binding,
        normal_sampler_layout_binding, metallic_sampler_layout_binding,
        roughness_sampler_layout_binding};

    const VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    VKH_CHECK(vkCreateDescriptorSetLayout(device_.device(), &layout_info,
                                          nullptr, &descriptor_set_layout_));
  }

  void create_graphics_pipeline()
  {
    const vkh::UniqueShaderModule vert_shader_module =
        vkh::create_unique_shader_module(device_.device(),
                                         "shaders/shader.vert.spv")
            .value();
    const vkh::UniqueShaderModule frag_shader_module =
        vkh::create_unique_shader_module(device_.device(),
                                         "shaders/shader.frag.spv")
            .value();

    const VkPipelineShaderStageCreateInfo vert_shader_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vert_shader_module.get(),
        .pName = "main",
    };

    const VkPipelineShaderStageCreateInfo frag_shader_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = frag_shader_module.get(),
        .pName = "main",
    };

    const std::array shader_stages{vert_shader_stage_info,
                                   frag_shader_stage_info};

    const auto binding_description = Vertex::get_binding_description();
    const auto attribute_descriptions = Vertex::get_attribute_descriptions();

    const VkPipelineVertexInputStateCreateInfo vertex_input_stage_create_info =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding_description,
            .vertexAttributeDescriptionCount =
                static_cast<uint32_t>(attribute_descriptions.size()),
            .pVertexAttributeDescriptions = attribute_descriptions.data(),
        };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        .primitiveRestartEnable = VK_FALSE};

    const VkViewport viewport = {
        .x = 0.0f,
        .y = static_cast<float>(swapchain_.extent().height),
        .width = static_cast<float>(swapchain_.extent().width),
        .height = -static_cast<float>(swapchain_.extent().height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const VkRect2D scissor = {.offset = {0, 0}, .extent = swapchain_.extent()};

    const VkPipelineViewportStateCreateInfo viewport_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    const VkPipelineRasterizationStateCreateInfo rasterizer_state_create_info =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
        };

    const VkPipelineMultisampleStateCreateInfo multisampling_state_create_info =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = device_.msaa_sample_count(),
            .sampleShadingEnable = VK_FALSE,
        };

    const VkPipelineDepthStencilStateCreateInfo
        depth_stencil_state_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
        };

    const VkPipelineColorBlendAttachmentState color_blend_attachment_state = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blending_state_create_info =
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
         .logicOpEnable = VK_FALSE,
         .logicOp = VK_LOGIC_OP_COPY,
         .attachmentCount = 1,
         .pAttachments = &color_blend_attachment_state,
         .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}};

    const VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout_,
    };

    VKH_CHECK(vkCreatePipelineLayout(device_.device(), &pipeline_layout_info,
                                     nullptr, &pipeline_layout_));

    const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = static_cast<uint32_t>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input_stage_create_info,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state_create_info,
        .pRasterizationState = &rasterizer_state_create_info,
        .pMultisampleState = &multisampling_state_create_info,
        .pDepthStencilState = &depth_stencil_state_create_info,
        .pColorBlendState = &color_blending_state_create_info,
        .layout = pipeline_layout_,
        .renderPass = render_pass_,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
    };

    VKH_CHECK(vkCreateGraphicsPipelines(device_.device(), VK_NULL_HANDLE, 1,
                                        &pipeline_info, nullptr,
                                        &graphics_pipeline_));
  }

  void create_framebuffers()
  {
    swapchain_framebuffers_.resize(swapchain_.image_views().size());

    for (size_t i = 0; i < swapchain_.image_views().size(); i++) {
      const std::array attachments = {color_image_view_, depth_image_view_,
                                      swapchain_.image_views()[i]};

      const VkFramebufferCreateInfo framebuffer_create_info = {
          .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
          .renderPass = render_pass_,
          .attachmentCount = static_cast<uint32_t>(attachments.size()),
          .pAttachments = attachments.data(),
          .width = swapchain_.extent().width,
          .height = swapchain_.extent().height,
          .layers = 1,
      };

      VKH_CHECK(vkCreateFramebuffer(device_.device(), &framebuffer_create_info,
                                    nullptr, &swapchain_framebuffers_[i]));
    }
  }

  void create_command_pool()
  {
    const vkh::QueueFamilyIndices queue_family_indices =
        device_.queue_family_indices();

    const VkCommandPoolCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = queue_family_indices.graphics_family,
    };

    VKH_CHECK(vkCreateCommandPool(device_.device(), &create_info, nullptr,
                                  &graphics_command_pool_));
  }

  void create_color_resources()
  {
    VkFormat color_format = swapchain_.image_format();

    create_image(swapchain_.extent().width, swapchain_.extent().height, 1,
                 device_.msaa_sample_count(), color_format,
                 VK_IMAGE_TILING_OPTIMAL,
                 VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                 color_image_, color_image_memory_);
    color_image_view_ = create_image_view(color_image_, color_format,
                                          VK_IMAGE_ASPECT_COLOR_BIT, 1);
  }

  void create_depth_resources()
  {
    const VkFormat depth_format = find_depth_format();
    create_image(swapchain_.extent().width, swapchain_.extent().height, 1,
                 device_.msaa_sample_count(), depth_format,
                 VK_IMAGE_TILING_OPTIMAL,
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depth_image_,
                 depth_image_memory_);
    depth_image_view_ = create_image_view(depth_image_, depth_format,
                                          VK_IMAGE_ASPECT_DEPTH_BIT, 1);
  }

  auto find_supported_format(const std::vector<VkFormat>& candidates,
                             VkImageTiling tiling,
                             VkFormatFeatureFlags features) -> VkFormat
  {
    for (VkFormat format : candidates) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(device_.vk_physical_device(), format,
                                          &props);

      if ((tiling == VK_IMAGE_TILING_LINEAR &&
           (props.linearTilingFeatures & features) == features) ||
          (tiling == VK_IMAGE_TILING_OPTIMAL &&
           (props.optimalTilingFeatures & features) == features)) {
        return format;
      }
    }

    beyond::panic("failed to find supported format!");
  }

  auto find_depth_format() -> VkFormat
  {
    return find_supported_format(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
         VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  }

  [[nodiscard]] auto create_texture_image(const char* texture_path) -> Texture
  {
    Texture texture;

    int tex_width, tex_height, tex_channels;

    stbi_uc* pixels = stbi_load(texture_path, &tex_width, &tex_height,
                                &tex_channels, STBI_rgb_alpha);
    const auto image_size =
        static_cast<VkDeviceSize>(tex_width * tex_height * 4);
    texture.mip_levels = static_cast<uint32_t>(std::floor(
                             std::log2(std::max(tex_width, tex_height)))) +
                         1;

    if (!pixels) {
      beyond::panic(
          fmt::format("failed to load texture image at {}!", texture_path));
    }

    auto staging_buffer =
        vkh::create_unique_buffer(device_.allocator(), image_size,
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VMA_MEMORY_USAGE_CPU_TO_GPU)
            .value();

    void* data = nullptr;
    staging_buffer.map(&data);
    memcpy(data, pixels, image_size);
    staging_buffer.unmap();

    stbi_image_free(pixels);

    auto texture_image_res = vkh::create_unique_image(
        device_.allocator(),
        vkh::ImageCreateInfo{
            .extent = {static_cast<uint32_t>(tex_width),
                       static_cast<uint32_t>(tex_height), 1},
            .mip_levels = texture.mip_levels,
            .samples_count = VK_SAMPLE_COUNT_1_BIT,
            .format = VK_FORMAT_R8G8B8A8_SRGB,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
        },
        VMA_MEMORY_USAGE_GPU_ONLY);
    texture.image = std::move(texture_image_res).value();

    transition_image_layout(
        texture.image.get(), VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, texture.mip_levels);
    copy_buffer_to_image(staging_buffer.get(), texture.image.get(),
                         static_cast<uint32_t>(tex_width),
                         static_cast<uint32_t>(tex_height));
    // transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating
    // mipmaps

    generate_mipmaps(texture.image.get(), VK_FORMAT_R8G8B8A8_SRGB, tex_width,
                     tex_height, texture.mip_levels);

    texture.image_view =
        create_image_view(texture.image.get(), VK_FORMAT_R8G8B8A8_SRGB,
                          VK_IMAGE_ASPECT_COLOR_BIT, texture.mip_levels);

    const VkSamplerCreateInfo sampler_create_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias = 0,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = 16,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0,
        .maxLod = static_cast<float>(texture.mip_levels),
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    VKH_CHECK(vkCreateSampler(device_.device(), &sampler_create_info, nullptr,
                              &texture.sampler));

    return texture;
  }

  void generate_mipmaps(VkImage image, VkFormat imageFormat, int32_t tex_width,
                        int32_t tex_height, uint32_t mip_levels)
  {
    // Check if image format supports linear blitting
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(device_.vk_physical_device(),
                                        imageFormat, &formatProperties);

    if (!(formatProperties.optimalTilingFeatures &
          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
      beyond::panic("texture image format does not support linear blitting!");
    }

    vkh::execute_single_time_command(
        device_.device(), graphics_command_pool_, device_.graphics_queue(),
        [&](VkCommandBuffer command_buffer) {
          VkImageMemoryBarrier barrier = {
              .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .image = image,
              .subresourceRange = {
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .levelCount = 1,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              }};

          int32_t mip_width = tex_width;
          int32_t mip_height = tex_height;

          for (uint32_t i = 1; i < mip_levels; ++i) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &barrier);

            const VkImageBlit blit = {
                .srcSubresource{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = i - 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .srcOffsets = {{0, 0, 0}, {mip_width, mip_height, 1}},
                .dstSubresource{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = i,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .dstOffsets = {{0, 0, 0},
                               {mip_width > 1 ? mip_width / 2 : 1,
                                mip_height > 1 ? mip_height / 2 : 1, 1}}};

            vkCmdBlitImage(command_buffer, image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                           VK_FILTER_LINEAR);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &barrier);

            if (mip_width > 1) mip_width /= 2;
            if (mip_height > 1) mip_height /= 2;
          }

          barrier.subresourceRange.baseMipLevel = mip_levels - 1;
          barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
          barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

          vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                               nullptr, 0, nullptr, 1, &barrier);
        });
  }

  VkImageView create_image_view(VkImage image, VkFormat format,
                                VkImageAspectFlags aspectFlags,
                                uint32_t mipLevels)
  {
    const VkImageViewCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspectFlags,
            .baseMipLevel = 0,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }};

    VkImageView image_view;
    VKH_CHECK(vkCreateImageView(device_.device(), &create_info, nullptr,
                                &image_view));

    return image_view;
  }

  void create_image(uint32_t width, uint32_t height, uint32_t mipLevels,
                    VkSampleCountFlagBits numSamples, VkFormat format,
                    VkImageTiling tiling, VkImageUsageFlags usage,
                    VkImage& image, VmaAllocation& imageMemory)
  {
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = numSamples;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info{.usage = VMA_MEMORY_USAGE_GPU_ONLY};
    VKH_CHECK(vmaCreateImage(device_.allocator(), &imageInfo, &alloc_info,
                             &image, &imageMemory, nullptr));
  }

  void transition_image_layout(VkImage image, VkFormat /*format*/,
                               VkImageLayout old_layout,
                               VkImageLayout new_layout, uint32_t mip_levels)
  {
    vkh::execute_single_time_command(
        device_.device(), graphics_command_pool_, device_.graphics_queue(),
        [&](VkCommandBuffer command_buffer) {
          VkImageMemoryBarrier barrier = {
              .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
              .oldLayout = old_layout,
              .newLayout = new_layout,
              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .image = image,
              .subresourceRange{
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .baseMipLevel = 0,
                  .levelCount = mip_levels,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              }};

          VkPipelineStageFlags source_stage;
          VkPipelineStageFlags destination_stage;

          if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
              new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
          } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                     new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
          } else {
            beyond::panic("unsupported layout transition!");
          }

          vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage,
                               0, 0, nullptr, 0, nullptr, 1, &barrier);
        });
  }

  void copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width,
                            uint32_t height)
  {
    vkh::execute_single_time_command(
        device_.device(), graphics_command_pool_, device_.graphics_queue(),
        [&](VkCommandBuffer command_buffer) {
          const VkBufferImageCopy region = {
              .bufferOffset = 0,
              .bufferRowLength = 0,
              .bufferImageHeight = 0,
              .imageSubresource =
                  {
                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                      .mipLevel = 0,
                      .baseArrayLayer = 0,
                      .layerCount = 1,
                  },
              .imageOffset = {0, 0, 0},
              .imageExtent = {width, height, 1},
          };

          vkCmdCopyBufferToImage(command_buffer, buffer, image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                 &region);
        });
  }

  void create_uniform_buffers()
  {
    constexpr VkDeviceSize buffer_size = sizeof(UniformBufferObject);

    uniform_buffers_.reserve(swapchain_.images().size());

    for (size_t i = 0; i < swapchain_.images().size(); i++) {
      uniform_buffers_.push_back(
          vkh::create_unique_buffer(device_.allocator(), buffer_size,
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    VMA_MEMORY_USAGE_CPU_TO_GPU)
              .value());
    }
  }

  void create_descriptor_pool()
  {
    const std::array pool_sizes = {
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount =
                static_cast<uint32_t>(swapchain_.images().size()),
        },
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount =
                static_cast<uint32_t>(swapchain_.images().size()) * 4,
        }};

    const VkDescriptorPoolCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = static_cast<uint32_t>(swapchain_.images().size()),
        .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    };

    VKH_CHECK(vkCreateDescriptorPool(device_.device(), &create_info, nullptr,
                                     &descriptor_pool_));
  }

  void create_descriptor_sets()
  {
    std::vector<VkDescriptorSetLayout> layouts(swapchain_.images().size(),
                                               descriptor_set_layout_);
    const VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool_,
        .descriptorSetCount = static_cast<uint32_t>(swapchain_.images().size()),
        .pSetLayouts = layouts.data(),
    };

    descriptor_sets_.resize(swapchain_.images().size());
    VKH_CHECK(vkAllocateDescriptorSets(device_.device(), &alloc_info,
                                       descriptor_sets_.data()));

    for (size_t i = 0; i < swapchain_.images().size(); i++) {
      const VkDescriptorBufferInfo buffer_info = {
          .buffer = uniform_buffers_[i].get(),
          .offset = 0,
          .range = sizeof(UniformBufferObject),
      };

      const VkDescriptorImageInfo albedo_texture_image_info = {
          .sampler = albedo_texture_.sampler,
          .imageView = albedo_texture_.image_view,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };

      const VkDescriptorImageInfo normal_texture_image_info = {
          .sampler = normal_texture_.sampler,
          .imageView = normal_texture_.image_view,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };

      const VkDescriptorImageInfo metallic_texture_image_info = {
          .sampler = metallic_texture_.sampler,
          .imageView = metallic_texture_.image_view,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };

      const VkDescriptorImageInfo roughness_texture_image_info = {
          .sampler = roughness_texture_.sampler,
          .imageView = roughness_texture_.image_view,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };

      const std::array write_descriptor_set = {
          VkWriteDescriptorSet{
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = descriptor_sets_[i],
              .dstBinding = 0,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              .pBufferInfo = &buffer_info,
          },
          VkWriteDescriptorSet{
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = descriptor_sets_[i],
              .dstBinding = 1,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .pImageInfo = &albedo_texture_image_info,
          },
          VkWriteDescriptorSet{
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = descriptor_sets_[i],
              .dstBinding = 2,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .pImageInfo = &normal_texture_image_info,
          },
          VkWriteDescriptorSet{
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = descriptor_sets_[i],
              .dstBinding = 3,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .pImageInfo = &metallic_texture_image_info,
          },
          VkWriteDescriptorSet{
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = descriptor_sets_[i],
              .dstBinding = 4,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .pImageInfo = &roughness_texture_image_info,
          }};

      vkUpdateDescriptorSets(device_.device(),
                             static_cast<uint32_t>(write_descriptor_set.size()),
                             write_descriptor_set.data(), 0, nullptr);
    }
  }

  auto find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties)
      -> uint32_t
  {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device_.vk_physical_device(),
                                        &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if (((typeFilter & (1u << i)) != 0u) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }

    beyond::panic("failed to find suitable memory type!");
  }

  void create_command_buffers()
  {
    command_buffers_.resize(swapchain_framebuffers_.size());

    const VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = graphics_command_pool_,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = static_cast<uint32_t>(command_buffers_.size()),
    };

    VKH_CHECK(vkAllocateCommandBuffers(device_.device(), &alloc_info,
                                       command_buffers_.data()));

    for (size_t i = 0; i < command_buffers_.size(); i++) {
      VkCommandBufferBeginInfo begin_info = {
          .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

      VKH_CHECK(vkBeginCommandBuffer(command_buffers_[i], &begin_info));

      constexpr std::array clear_values = {
          VkClearValue{.color = {{0.0f, 0.0f, 0.0f, 1.0f}}},
          VkClearValue{.depthStencil = {1.0f, 0}}};

      const VkRenderPassBeginInfo render_pass_begin_info = {
          .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
          .renderPass = render_pass_,
          .framebuffer = swapchain_framebuffers_[i],
          .renderArea{
              .offset = {0, 0},
              .extent = swapchain_.extent(),
          },
          .clearValueCount = static_cast<uint32_t>(clear_values.size()),
          .pClearValues = clear_values.data(),
      };

      vkCmdBeginRenderPass(command_buffers_[i], &render_pass_begin_info,
                           VK_SUBPASS_CONTENTS_INLINE);

      vkCmdBindPipeline(command_buffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphics_pipeline_);

      VkBuffer vertex_buffers[] = {mesh_.vertex_buffer.get()};
      VkDeviceSize offsets[] = {0};
      vkCmdBindVertexBuffers(command_buffers_[i], 0, 1, vertex_buffers,
                             offsets);

      vkCmdBindIndexBuffer(command_buffers_[i], mesh_.index_buffer.get(), 0,
                           VK_INDEX_TYPE_UINT32);

      vkCmdBindDescriptorSets(command_buffers_[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                              0, 1, &descriptor_sets_[i], 0, nullptr);

      vkCmdDrawIndexed(command_buffers_[i], mesh_.indices_size, 1, 0, 0, 0);
      vkCmdEndRenderPass(command_buffers_[i]);

      VKH_CHECK(vkEndCommandBuffer(command_buffers_[i]));
    }
  }

  void create_sync_objects()
  {
    image_available_semaphores_.resize(max_frames_in_flight);
    render_finished_semaphores_.resize(max_frames_in_flight);
    in_flight_fences_.resize(max_frames_in_flight);
    images_in_flight_.resize(swapchain_.images().size(), VK_NULL_HANDLE);

    const VkSemaphoreCreateInfo semaphore_create_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    const VkFenceCreateInfo fence_create_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    for (size_t i = 0; i < max_frames_in_flight; i++) {
      if (vkCreateSemaphore(device_.device(), &semaphore_create_info, nullptr,
                            &image_available_semaphores_[i]) != VK_SUCCESS ||
          vkCreateSemaphore(device_.device(), &semaphore_create_info, nullptr,
                            &render_finished_semaphores_[i]) != VK_SUCCESS ||
          vkCreateFence(device_.device(), &fence_create_info, nullptr,
                        &in_flight_fences_[i]) != VK_SUCCESS) {
        beyond::panic("failed to create synchronization objects for a frame!");
      }
    }
  }

  void update_uniform_buffer(uint32_t current_image)
  {
    const UniformBufferObject ubo = {
        .model = glm::rotate(glm::mat4(1.0f), rotation_z_.value(),
                             glm::vec3(0.0f, 0.0f, 1.0f)) *
                 glm::rotate(glm::mat4(1.0f), rotation_x_.value(),
                             glm::vec3(1.0f, 0.0f, 0.0f)),
        .view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),
                            glm::vec3(0.0f, 0.0f, 0.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f)),
        .proj =
            glm::perspective(glm::radians(45.0f),
                             static_cast<float>(swapchain_.extent().width) /
                                 static_cast<float>(swapchain_.extent().height),
                             0.1f, 10.0f),
    };

    void* data = nullptr;
    uniform_buffers_[current_image].map(&data);
    memcpy(data, &ubo, sizeof(ubo));
    uniform_buffers_[current_image].unmap();
  }

  void draw_frame()
  {
    vkWaitForFences(device_.device(), 1, &in_flight_fences_[current_frame_],
                    VK_TRUE, UINT64_MAX);

    uint32_t image_index = 0;
    VkResult result =
        vkAcquireNextImageKHR(device_.device(), swapchain_, UINT64_MAX,
                              image_available_semaphores_[current_frame_],
                              VK_NULL_HANDLE, &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      recreate_swapchain();
      return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      beyond::panic("failed to acquire swap chain image!");
    }

    update_uniform_buffer(image_index);

    if (images_in_flight_[image_index] != VK_NULL_HANDLE) {
      vkWaitForFences(device_.device(), 1, &images_in_flight_[image_index],
                      VK_TRUE, UINT64_MAX);
    }
    images_in_flight_[image_index] = in_flight_fences_[current_frame_];

    std::array wait_semaphores = {image_available_semaphores_[current_frame_]};
    std::array signal_semaphores = {
        render_finished_semaphores_[current_frame_]};
    VkPipelineStageFlags wait_stages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    const VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = static_cast<uint32_t>(wait_semaphores.size()),
        .pWaitSemaphores = wait_semaphores.data(),
        .pWaitDstStageMask = wait_stages,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffers_[image_index],
        .signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size()),
        .pSignalSemaphores = signal_semaphores.data(),
    };

    vkResetFences(device_.device(), 1, &in_flight_fences_[current_frame_]);

    VKH_CHECK(vkQueueSubmit(device_.graphics_queue(), 1, &submit_info,
                            in_flight_fences_[current_frame_]));

    const std::array swapchains{swapchain_.get()};

    const VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size()),
        .pWaitSemaphores = signal_semaphores.data(),
        .swapchainCount = static_cast<uint32_t>(swapchains.size()),
        .pSwapchains = swapchains.data(),
        .pImageIndices = &image_index,
    };

    result = vkQueuePresentKHR(device_.present_queue(), &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
        framebuffer_resized_) {
      framebuffer_resized_ = false;
      recreate_swapchain();
    } else if (result != VK_SUCCESS) {
      beyond::panic("failed to present swap chain image!");
    }

    current_frame_ = (current_frame_ + 1) % max_frames_in_flight;
  }
};

auto main() -> int
{
  Application app;

  try {
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
