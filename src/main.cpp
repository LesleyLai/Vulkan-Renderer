#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "model.hpp"
#include "texture.hpp"

#include "vulkan_helper/buffer.hpp"
#include "vulkan_helper/check.hpp"
#include "vulkan_helper/gpu_device.hpp"
#include "vulkan_helper/shader_module.hpp"
#include "vulkan_helper/single_time_command.hpp"
#include "vulkan_helper/swapchain.hpp"

#include "vulkan_helper/image.hpp"
#include "vulkan_helper/image_view.hpp"

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

constexpr int max_frames_in_flight = 2;

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
  glm::vec3 camera_pos;
};

auto find_supported_format(VkPhysicalDevice pd,
                           const std::vector<VkFormat>& candidates,
                           VkImageTiling tiling, VkFormatFeatureFlags features)
    -> VkFormat
{
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(pd, format, &props);

    if ((tiling == VK_IMAGE_TILING_LINEAR &&
         (props.linearTilingFeatures & features) == features) ||
        (tiling == VK_IMAGE_TILING_OPTIMAL &&
         (props.optimalTilingFeatures & features) == features)) {
      return format;
    }
  }

  beyond::panic("failed to find supported format!");
}

auto find_depth_format(VkPhysicalDevice pd) -> VkFormat
{
  return find_supported_format(
      pd,
      {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
       VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

auto generate_uv_sphere(vkh::GPUDevice& device) -> StaticMesh
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

  return create_mesh_from_data(device, vertices, indices);
}

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

  vkh::UniqueImage color_image_{};
  vkh::UniqueImageView color_image_view_{};

  vkh::UniqueImage depth_image_{};
  vkh::UniqueImageView depth_image_view_{};

  Texture albedo_texture_{};
  Texture normal_texture_{};
  Texture metallic_texture_{};
  Texture roughness_texture_{};

  Model model_;
  // StaticMesh mesh_;
  std::vector<vkh::UniqueBuffer> uniform_buffers_;

  VkDescriptorPool descriptor_pool_{};
  std::vector<VkDescriptorSet> descriptor_sets_;

  std::vector<VkCommandBuffer> command_buffers_;

  std::vector<VkSemaphore> image_available_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> in_flight_fences_;
  std::vector<VkFence> images_in_flight_;
  size_t current_frame_ = 0;

  glm::vec3 camera_pos_{2.0f, 2.0f, 2.0f};
  glm::vec3 up_{0.0f, 0.0f, 1.0f};

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

    const glm::vec3 up{0.0f, 0.0f, 1.0f};

    [[maybe_unused]] auto* app =
        beyond::bit_cast<Application*>(glfwGetWindowUserPointer(window));

    switch (action) {
    case GLFW_PRESS:
      [[fallthrough]];
    case GLFW_REPEAT:
      switch (key) {
      case GLFW_KEY_W: {
        const auto axis =
            glm::normalize(glm::cross(app->camera_pos_, app->up_));
        app->camera_pos_ =
            glm::rotate(app->camera_pos_, (0.1_rad).value(), axis);
        app->up_ = glm::rotate(app->up_, (0.1_rad).value(), axis);
      } break;
      case GLFW_KEY_S: {
        const auto axis =
            glm::normalize(glm::cross(app->camera_pos_, app->up_));
        app->camera_pos_ =
            glm::rotate(app->camera_pos_, (-0.1_rad).value(), axis);
        app->up_ = glm::rotate(app->up_, (-0.1_rad).value(), axis);
      } break;
      case GLFW_KEY_A:
        app->camera_pos_ =
            glm::rotate(app->camera_pos_, (0.1_rad).value(), app->up_);
        break;
      case GLFW_KEY_D:
        app->camera_pos_ =
            glm::rotate(app->camera_pos_, (-0.1_rad).value(), app->up_);
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

    create_color_resources();
    create_depth_resources();
    create_framebuffers();

    albedo_texture_ = create_texture_from_file(
        device_, "textures/rustediron1-alt2-bl/rustediron2_basecolor.png");
    normal_texture_ = create_texture_from_file(
        device_, "textures/rustediron1-alt2-bl/rustediron2_normal.png");
    metallic_texture_ = create_texture_from_file(
        device_, "textures/rustediron1-alt2-bl/rustediron2_metallic.png");
    roughness_texture_ = create_texture_from_file(
        device_, "textures/rustediron1-alt2-bl/rustediron2_roughness.png");

    model_ = Model::load(device_, "models/DamagedHelmet.gltf");

    std::cout << model_.meshes().size() << std::endl;

    create_uniform_buffers();
    create_descriptor_pool();
    create_descriptor_sets();
    create_command_buffers();
    create_sync_objects();
  }

  void cleanup_swapchain()
  {
    depth_image_view_.reset();
    depth_image_.reset();
    color_image_view_.reset();
    color_image_.reset();

    for (auto* framebuffer : swapchain_framebuffers_) {
      vkDestroyFramebuffer(device_.device(), framebuffer, nullptr);
    }

    swapchain_.reset();

    vkFreeCommandBuffers(device_.device(), device_.graphics_command_pool(),
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

    destroy_texture(device_, roughness_texture_);
    destroy_texture(device_, metallic_texture_);
    destroy_texture(device_, normal_texture_);
    destroy_texture(device_, albedo_texture_);

    vkDestroyDescriptorSetLayout(device_.device(), descriptor_set_layout_,
                                 nullptr);

    for (size_t i = 0; i < max_frames_in_flight; i++) {
      vkDestroySemaphore(device_.device(), render_finished_semaphores_[i],
                         nullptr);
      vkDestroySemaphore(device_.device(), image_available_semaphores_[i],
                         nullptr);
      vkDestroyFence(device_.device(), in_flight_fences_[i], nullptr);
    }
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
        .format = find_depth_format(device_.vk_physical_device()),
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
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
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
      const std::array attachments = {color_image_view_.get(),
                                      depth_image_view_.get(),
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

  void create_framebuffer_attachment_resource(std::string image_name,
                                              VkFormat image_format,
                                              VkImageUsageFlags image_usage,
                                              VkImageAspectFlags image_aspect,
                                              vkh::UniqueImage& image,
                                              vkh::UniqueImageView& image_view)
  {
    auto image_ret =
        vkh::create_unique_image(device_.allocator(),
                                 {.extent =
                                      {
                                          swapchain_.extent().width,
                                          swapchain_.extent().height,
                                          1,
                                      },
                                  .mip_levels = 1,
                                  .samples_count = device_.msaa_sample_count(),
                                  .format = image_format,
                                  .usage = image_usage},
                                 VMA_MEMORY_USAGE_GPU_ONLY);
    if (!image_ret.has_value()) {
      beyond::panic("Cannot create {} image resource!", image_name);
    }
    image = std::move(image_ret).value();

    auto image_view_ret = vkh::create_unique_image_view(
        device_.device(), {.image = image.get(),
                           .view_type = VK_IMAGE_VIEW_TYPE_2D,
                           .format = image_format,
                           .subresource_range = {
                               .aspectMask = image_aspect,
                               .baseMipLevel = 0,
                               .levelCount = 1,
                               .baseArrayLayer = 0,
                               .layerCount = 1,
                           }});
    if (!image_view_ret.has_value()) {
      beyond::panic("Cannot create {} image view resource!", image_name);
    }
    image_view = std::move(image_view_ret).value();
  }

  void create_color_resources()
  {
    create_framebuffer_attachment_resource(
        "color", swapchain_.image_format(),
        VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, color_image_, color_image_view_);
  }

  void create_depth_resources()
  {
    create_framebuffer_attachment_resource(
        "depth", find_depth_format(device_.vk_physical_device()),
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
        depth_image_, depth_image_view_);
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
        .commandPool = device_.graphics_command_pool(),
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = static_cast<uint32_t>(command_buffers_.size()),
    };

    VKH_CHECK(vkAllocateCommandBuffers(device_.device(), &alloc_info,
                                       command_buffers_.data()));

    for (size_t i = 0; i < command_buffers_.size(); ++i) {
      auto* command_buffer = command_buffers_[i];

      VkCommandBufferBeginInfo begin_info = {
          .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

      VKH_CHECK(vkBeginCommandBuffer(command_buffer, &begin_info));

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

      vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info,
                           VK_SUBPASS_CONTENTS_INLINE);

      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphics_pipeline_);

      const auto& mesh = model_.meshes()[0];

      VkBuffer vertex_buffers[] = {mesh.vertex_buffer.get()};
      VkDeviceSize offsets[] = {0};
      vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);

      vkCmdBindIndexBuffer(command_buffer, mesh.index_buffer.get(), 0,
                           VK_INDEX_TYPE_UINT32);

      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline_layout_, 0, 1, &descriptor_sets_[i], 0,
                              nullptr);

      vkCmdDrawIndexed(command_buffer, mesh.indices_size, 1, 0, 0, 0);
      vkCmdEndRenderPass(command_buffer);

      VKH_CHECK(vkEndCommandBuffer(command_buffer));
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
        .model = glm::mat4(1.0f),
        .view = glm::lookAt(camera_pos_, glm::vec3(0.0f, 0.0f, 0.0f), up_),
        .proj =
            glm::perspective(glm::radians(45.0f),
                             static_cast<float>(swapchain_.extent().width) /
                                 static_cast<float>(swapchain_.extent().height),
                             0.1f, 10.0f),
        .camera_pos = camera_pos_};

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
