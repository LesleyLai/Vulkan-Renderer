#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "wrappers/gpu_device.hpp"
#include "wrappers/shader_module.hpp"
#include "wrappers/window.hpp"

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
#include <unordered_map>
#include <vector>

constexpr int init_width = 800;
constexpr int init_height = 600;

constexpr const char* model_path = "models/chalet.obj";
constexpr const char* texture_path = "textures/chalet.jpg";

constexpr int max_frames_in_flight = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

constexpr bool enableValidationLayers = true;

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger)
{
  auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator)
{
  auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  bool isComplete()
  {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static VkVertexInputBindingDescription getBindingDescription()
  {
    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 3>
  getAttributeDescriptions()
  {
    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

    return attributeDescriptions;
  }

  bool operator==(const Vertex& other) const
  {
    return pos == other.pos && color == other.color &&
           texCoord == other.texCoord;
  }
};

namespace std {
template <> struct hash<Vertex> {
  size_t operator()(Vertex const& vertex) const
  {
    return ((hash<glm::vec3>()(vertex.pos) ^
             (hash<glm::vec3>()(vertex.color) << 1)) >>
            1) ^
           (hash<glm::vec2>()(vertex.texCoord) << 1);
  }
};
} // namespace std

struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

class HelloTriangleApplication {
public:
  HelloTriangleApplication()
      : window_{init_width, init_height, "Vulkan"}, device_{window_}
  {
    glfwSetWindowUserPointer(window_.get(), this);
    glfwSetFramebufferSizeCallback(window_.get(), framebufferResizeCallback);

    initVulkan();
  }

  void run()
  {
    main_loop();
    cleanup();
  }

private:
  Window window_;
  GPUDevice device_;

  VkSwapchainKHR swapchain_;
  std::vector<VkImage> swapchain_images_;
  VkFormat swapchain_image_format_;
  VkExtent2D swapChainExtent;
  std::vector<VkImageView> swapChainImageViews;
  std::vector<VkFramebuffer> swapChainFramebuffers;

  VkRenderPass render_pass_;
  VkDescriptorSetLayout descriptor_set_layout_;
  VkPipelineLayout pipeline_layout_;
  VkPipeline graphics_pipeline_;

  VkCommandPool command_pool_;

  VkImage color_image_;
  VkDeviceMemory color_image_memory_;
  VkImageView color_image_view_;

  VkImage depth_image_;
  VkDeviceMemory depth_image_memory_;
  VkImageView depth_image_view_;

  uint32_t mip_levels_;
  VkImage texture_image_;
  VkDeviceMemory texture_image_memory_;
  VkImageView texture_image_view_;
  VkSampler texture_sampler_;

  std::vector<Vertex> vertices_;
  std::vector<uint32_t> indices_;
  VkBuffer vertex_buffer_;
  VkDeviceMemory vertex_buffer_memory_;
  VkBuffer index_buffer_;
  VkDeviceMemory index_buffer_memory_;

  std::vector<VkBuffer> uniform_buffers_;
  std::vector<VkDeviceMemory> uniform_buffers_memory_;

  VkDescriptorPool descriptor_pool_;
  std::vector<VkDescriptorSet> descriptor_sets_;

  std::vector<VkCommandBuffer> command_buffers_;

  std::vector<VkSemaphore> image_available_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> in_flight_fences_;
  std::vector<VkFence> images_in_flight_;
  size_t current_frame_ = 0;

  bool framebuffer_resized = false;

  static void framebufferResizeCallback(GLFWwindow* window, int /*width*/,
                                        int /*height*/)
  {
    auto app = reinterpret_cast<HelloTriangleApplication*>(
        glfwGetWindowUserPointer(window));
    app->framebuffer_resized = true;
  }

  void initVulkan()
  {
    create_swapchain();
    create_image_views();
    create_render_pass();
    create_descriptor_set_layout();
    create_graphics_pipeline();
    create_command_pool();
    create_color_resources();
    create_depth_resources();
    create_framebuffers();
    create_texture_image();
    create_texture_image_view();
    create_texture_sampler();
    load_model();
    create_vertex_buffer();
    create_index_buffer();
    create_uniform_buffers();
    create_descriptor_pool();
    create_descriptor_sets();
    create_command_buffers();
    create_sync_objects();
  }

  void main_loop()
  {
    while (!window_.should_close()) {
      glfwPollEvents();
      drawFrame();
    }

    device_.wait_idle();
  }

  void cleanupSwapChain()
  {
    vkDestroyImageView(device_.vk_device(), depth_image_view_, nullptr);
    vkDestroyImage(device_.vk_device(), depth_image_, nullptr);
    vkFreeMemory(device_.vk_device(), depth_image_memory_, nullptr);

    vkDestroyImageView(device_.vk_device(), color_image_view_, nullptr);
    vkDestroyImage(device_.vk_device(), color_image_, nullptr);
    vkFreeMemory(device_.vk_device(), color_image_memory_, nullptr);

    for (auto framebuffer : swapChainFramebuffers) {
      vkDestroyFramebuffer(device_.vk_device(), framebuffer, nullptr);
    }

    vkFreeCommandBuffers(device_.vk_device(), command_pool_,
                         static_cast<uint32_t>(command_buffers_.size()),
                         command_buffers_.data());

    vkDestroyPipeline(device_.vk_device(), graphics_pipeline_, nullptr);
    vkDestroyPipelineLayout(device_.vk_device(), pipeline_layout_, nullptr);
    vkDestroyRenderPass(device_.vk_device(), render_pass_, nullptr);

    for (auto imageView : swapChainImageViews) {
      vkDestroyImageView(device_.vk_device(), imageView, nullptr);
    }

    vkDestroySwapchainKHR(device_.vk_device(), swapchain_, nullptr);

    for (size_t i = 0; i < swapchain_images_.size(); i++) {
      vkDestroyBuffer(device_.vk_device(), uniform_buffers_[i], nullptr);
      vkFreeMemory(device_.vk_device(), uniform_buffers_memory_[i], nullptr);
    }

    vkDestroyDescriptorPool(device_.vk_device(), descriptor_pool_, nullptr);
  }

  void cleanup()
  {
    cleanupSwapChain();

    vkDestroySampler(device_.vk_device(), texture_sampler_, nullptr);
    vkDestroyImageView(device_.vk_device(), texture_image_view_, nullptr);

    vkDestroyImage(device_.vk_device(), texture_image_, nullptr);
    vkFreeMemory(device_.vk_device(), texture_image_memory_, nullptr);

    vkDestroyDescriptorSetLayout(device_.vk_device(), descriptor_set_layout_,
                                 nullptr);

    vkDestroyBuffer(device_.vk_device(), index_buffer_, nullptr);
    vkFreeMemory(device_.vk_device(), index_buffer_memory_, nullptr);

    vkDestroyBuffer(device_.vk_device(), vertex_buffer_, nullptr);
    vkFreeMemory(device_.vk_device(), vertex_buffer_memory_, nullptr);

    for (size_t i = 0; i < max_frames_in_flight; i++) {
      vkDestroySemaphore(device_.vk_device(), render_finished_semaphores_[i],
                         nullptr);
      vkDestroySemaphore(device_.vk_device(), image_available_semaphores_[i],
                         nullptr);
      vkDestroyFence(device_.vk_device(), in_flight_fences_[i], nullptr);
    }

    vkDestroyCommandPool(device_.vk_device(), command_pool_, nullptr);
  }

  void recreateSwapChain()
  {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window_.get(), &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window_.get(), &width, &height);
      glfwWaitEvents();
    }

    vkDeviceWaitIdle(device_.vk_device());

    cleanupSwapChain();

    create_swapchain();
    create_image_views();
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

  void create_swapchain()
  {
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(device_.vk_physical_device());

    VkSurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = device_.surface();

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices inds = findQueueFamilies(device_.vk_physical_device());
    uint32_t queueFamilyIndices[] = {inds.graphicsFamily.value(),
                                     inds.presentFamily.value()};

    if (inds.graphicsFamily != inds.presentFamily) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device_.vk_device(), &createInfo, nullptr,
                             &swapchain_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device_.vk_device(), swapchain_, &imageCount,
                            nullptr);
    swapchain_images_.resize(imageCount);
    vkGetSwapchainImagesKHR(device_.vk_device(), swapchain_, &imageCount,
                            swapchain_images_.data());

    swapchain_image_format_ = surfaceFormat.format;
    swapChainExtent = extent;
  }

  void create_image_views()
  {
    swapChainImageViews.resize(swapchain_images_.size());

    for (uint32_t i = 0; i < swapchain_images_.size(); i++) {
      swapChainImageViews[i] =
          create_image_view(swapchain_images_[i], swapchain_image_format_,
                            VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }
  }

  void create_render_pass()
  {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = swapchain_image_format_;
    colorAttachment.samples = device_.msaa_sample_count();
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = find_depth_format();
    depthAttachment.samples = device_.msaa_sample_count();
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachmentResolve = {};
    colorAttachmentResolve.format = swapchain_image_format_;
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentResolveRef = {};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 3> attachments = {
        colorAttachment, depthAttachment, colorAttachmentResolve};
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device_.vk_device(), &renderPassInfo, nullptr,
                           &render_pass_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass!");
    }
  }

  void create_descriptor_set_layout()
  {
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        uboLayoutBinding, samplerLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device_.vk_device(), &layoutInfo, nullptr,
                                    &descriptor_set_layout_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor set layout!");
    }
  }

  void create_graphics_pipeline()
  {
    auto vertShaderCode = read_file("shaders/shader.vert.spv");
    auto fragShaderCode = read_file("shaders/shader.frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = device_.msaa_sample_count();

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptor_set_layout_;

    if (vkCreatePipelineLayout(device_.vk_device(), &pipelineLayoutInfo,
                               nullptr, &pipeline_layout_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipeline_layout_;
    pipelineInfo.renderPass = render_pass_;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(device_.vk_device(), VK_NULL_HANDLE, 1,
                                  &pipelineInfo, nullptr,
                                  &graphics_pipeline_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device_.vk_device(), fragShaderModule, nullptr);
    vkDestroyShaderModule(device_.vk_device(), vertShaderModule, nullptr);
  }

  void create_framebuffers()
  {
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      std::array<VkImageView, 3> attachments = {
          color_image_view_, depth_image_view_, swapChainImageViews[i]};

      VkFramebufferCreateInfo framebufferInfo = {};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferInfo.renderPass = render_pass_;
      framebufferInfo.attachmentCount =
          static_cast<uint32_t>(attachments.size());
      framebufferInfo.pAttachments = attachments.data();
      framebufferInfo.width = swapChainExtent.width;
      framebufferInfo.height = swapChainExtent.height;
      framebufferInfo.layers = 1;

      if (vkCreateFramebuffer(device_.vk_device(), &framebufferInfo, nullptr,
                              &swapChainFramebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
      }
    }
  }

  void create_command_pool()
  {
    QueueFamilyIndices queueFamilyIndices =
        findQueueFamilies(device_.vk_physical_device());

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if (vkCreateCommandPool(device_.vk_device(), &poolInfo, nullptr,
                            &command_pool_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics command pool!");
    }
  }

  void create_color_resources()
  {
    VkFormat colorFormat = swapchain_image_format_;

    create_image(
        swapChainExtent.width, swapChainExtent.height, 1,
        device_.msaa_sample_count(), colorFormat, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, color_image_, color_image_memory_);
    color_image_view_ = create_image_view(color_image_, colorFormat,
                                          VK_IMAGE_ASPECT_COLOR_BIT, 1);
  }

  void create_depth_resources()
  {
    VkFormat depthFormat = find_depth_format();

    create_image(
        swapChainExtent.width, swapChainExtent.height, 1,
        device_.msaa_sample_count(), depthFormat, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depth_image_, depth_image_memory_);
    depth_image_view_ = create_image_view(depth_image_, depthFormat,
                                          VK_IMAGE_ASPECT_DEPTH_BIT, 1);
  }

  VkFormat find_supported_format(const std::vector<VkFormat>& candidates,
                                 VkImageTiling tiling,
                                 VkFormatFeatureFlags features)
  {
    for (VkFormat format : candidates) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(device_.vk_physical_device(), format,
                                          &props);

      if (tiling == VK_IMAGE_TILING_LINEAR &&
          (props.linearTilingFeatures & features) == features) {
        return format;
      } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                 (props.optimalTilingFeatures & features) == features) {
        return format;
      }
    }

    throw std::runtime_error("failed to find supported format!");
  }

  VkFormat find_depth_format()
  {
    return find_supported_format(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
         VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  }

  void create_texture_image()
  {
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(texture_path, &texWidth, &texHeight,
                                &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize =
        static_cast<VkDeviceSize>(texWidth * texHeight * 4);
    mip_levels_ = static_cast<uint32_t>(
                      std::floor(std::log2(std::max(texWidth, texHeight)))) +
                  1;

    if (!pixels) {
      throw std::runtime_error("failed to load texture image!");
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    create_buffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device_.vk_device(), stagingBufferMemory, 0, imageSize, 0,
                &data);
    memcpy(data, pixels, imageSize);
    vkUnmapMemory(device_.vk_device(), stagingBufferMemory);

    stbi_image_free(pixels);

    create_image(
        static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight),
        mip_levels_, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texture_image_,
        texture_image_memory_);

    transition_image_layout(texture_image_, VK_FORMAT_R8G8B8A8_SRGB,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mip_levels_);
    copy_buffer_to_image(stagingBuffer, texture_image_,
                         static_cast<uint32_t>(texWidth),
                         static_cast<uint32_t>(texHeight));
    // transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating
    // mipmaps

    vkDestroyBuffer(device_.vk_device(), stagingBuffer, nullptr);
    vkFreeMemory(device_.vk_device(), stagingBufferMemory, nullptr);

    generate_mipmaps(texture_image_, VK_FORMAT_R8G8B8A8_SRGB, texWidth,
                     texHeight, mip_levels_);
  }

  void generate_mipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth,
                        int32_t texHeight, uint32_t mipLevels)
  {
    // Check if image format supports linear blitting
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(device_.vk_physical_device(),
                                        imageFormat, &formatProperties);

    if (!(formatProperties.optimalTilingFeatures &
          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
      throw std::runtime_error(
          "texture image format does not support linear blitting!");
    }

    VkCommandBuffer commandBuffer = begin_single_time_commands();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++) {
      barrier.subresourceRange.baseMipLevel = i - 1;
      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

      vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &barrier);

      VkImageBlit blit = {};
      blit.srcOffsets[0] = {0, 0, 0};
      blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
      blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.srcSubresource.mipLevel = i - 1;
      blit.srcSubresource.baseArrayLayer = 0;
      blit.srcSubresource.layerCount = 1;
      blit.dstOffsets[0] = {0, 0, 0};
      blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1,
                            mipHeight > 1 ? mipHeight / 2 : 1, 1};
      blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.dstSubresource.mipLevel = i;
      blit.dstSubresource.baseArrayLayer = 0;
      blit.dstSubresource.layerCount = 1;

      vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                     VK_FILTER_LINEAR);

      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, 1, &barrier);

      if (mipWidth > 1)
        mipWidth /= 2;
      if (mipHeight > 1)
        mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);

    end_single_time_commands(commandBuffer);
  }

  void create_texture_image_view()
  {
    texture_image_view_ =
        create_image_view(texture_image_, VK_FORMAT_R8G8B8A8_SRGB,
                          VK_IMAGE_ASPECT_COLOR_BIT, mip_levels_);
  }

  void create_texture_sampler()
  {
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0;
    samplerInfo.maxLod = static_cast<float>(mip_levels_);
    samplerInfo.mipLodBias = 0;

    if (vkCreateSampler(device_.vk_device(), &samplerInfo, nullptr,
                        &texture_sampler_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create texture sampler!");
    }
  }

  VkImageView create_image_view(VkImage image, VkFormat format,
                                VkImageAspectFlags aspectFlags,
                                uint32_t mipLevels)
  {
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device_.vk_device(), &viewInfo, nullptr,
                          &imageView) != VK_SUCCESS) {
      throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
  }

  void create_image(uint32_t width, uint32_t height, uint32_t mipLevels,
                    VkSampleCountFlagBits numSamples, VkFormat format,
                    VkImageTiling tiling, VkImageUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkImage& image,
                    VkDeviceMemory& imageMemory)
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

    if (vkCreateImage(device_.vk_device(), &imageInfo, nullptr, &image) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device_.vk_device(), image, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        find_memory_type(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device_.vk_device(), &allocInfo, nullptr,
                         &imageMemory) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device_.vk_device(), image, imageMemory, 0);
  }

  void transition_image_layout(VkImage image, VkFormat /*format*/,
                               VkImageLayout oldLayout, VkImageLayout newLayout,
                               uint32_t mipLevels)
  {
    VkCommandBuffer commandBuffer = begin_single_time_commands();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

      sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);

    end_single_time_commands(commandBuffer);
  }

  void copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width,
                            uint32_t height)
  {
    VkCommandBuffer commandBuffer = begin_single_time_commands();

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(commandBuffer, buffer, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    end_single_time_commands(commandBuffer);
  }

  void load_model()
  {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, model_path)) {
      throw std::runtime_error(err);
    }

    std::unordered_map<Vertex, uint32_t> uniqueVertices = {};

    for (const auto& shape : shapes) {
      for (const auto& index : shape.mesh.indices) {
        Vertex vertex = {};

        vertex.pos = {
            attrib.vertices[static_cast<size_t>(3 * index.vertex_index + 0)],
            attrib.vertices[static_cast<size_t>(3 * index.vertex_index + 1)],
            attrib.vertices[static_cast<size_t>(3 * index.vertex_index + 2)]};

        vertex.texCoord = {
            attrib.texcoords[static_cast<size_t>(2 * index.texcoord_index + 0)],
            1.0f - attrib.texcoords[static_cast<size_t>(
                       2 * index.texcoord_index + 1)]};

        vertex.color = {1.0f, 1.0f, 1.0f};

        if (uniqueVertices.count(vertex) == 0) {
          uniqueVertices[vertex] = static_cast<uint32_t>(vertices_.size());
          vertices_.push_back(vertex);
        }

        indices_.push_back(uniqueVertices[vertex]);
      }
    }
  }

  void create_vertex_buffer()
  {
    VkDeviceSize bufferSize = sizeof(vertices_[0]) * vertices_.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device_.vk_device(), stagingBufferMemory, 0, bufferSize, 0,
                &data);
    memcpy(data, vertices_.data(), bufferSize);
    vkUnmapMemory(device_.vk_device(), stagingBufferMemory);

    create_buffer(bufferSize,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer_,
                  vertex_buffer_memory_);

    copy_buffer(stagingBuffer, vertex_buffer_, bufferSize);

    vkDestroyBuffer(device_.vk_device(), stagingBuffer, nullptr);
    vkFreeMemory(device_.vk_device(), stagingBufferMemory, nullptr);
  }

  void create_index_buffer()
  {
    VkDeviceSize bufferSize = sizeof(indices_[0]) * indices_.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device_.vk_device(), stagingBufferMemory, 0, bufferSize, 0,
                &data);
    memcpy(data, indices_.data(), bufferSize);
    vkUnmapMemory(device_.vk_device(), stagingBufferMemory);

    create_buffer(bufferSize,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, index_buffer_,
                  index_buffer_memory_);

    copy_buffer(stagingBuffer, index_buffer_, bufferSize);

    vkDestroyBuffer(device_.vk_device(), stagingBuffer, nullptr);
    vkFreeMemory(device_.vk_device(), stagingBufferMemory, nullptr);
  }

  void create_uniform_buffers()
  {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniform_buffers_.resize(swapchain_images_.size());
    uniform_buffers_memory_.resize(swapchain_images_.size());

    for (size_t i = 0; i < swapchain_images_.size(); i++) {
      create_buffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    uniform_buffers_[i], uniform_buffers_memory_[i]);
    }
  }

  void create_descriptor_pool()
  {
    std::array<VkDescriptorPoolSize, 2> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount =
        static_cast<uint32_t>(swapchain_images_.size());
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount =
        static_cast<uint32_t>(swapchain_images_.size());

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(swapchain_images_.size());

    if (vkCreateDescriptorPool(device_.vk_device(), &poolInfo, nullptr,
                               &descriptor_pool_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor pool!");
    }
  }

  void create_descriptor_sets()
  {
    std::vector<VkDescriptorSetLayout> layouts(swapchain_images_.size(),
                                               descriptor_set_layout_);
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptor_pool_;
    allocInfo.descriptorSetCount =
        static_cast<uint32_t>(swapchain_images_.size());
    allocInfo.pSetLayouts = layouts.data();

    descriptor_sets_.resize(swapchain_images_.size());
    if (vkAllocateDescriptorSets(device_.vk_device(), &allocInfo,
                                 descriptor_sets_.data()) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < swapchain_images_.size(); i++) {
      VkDescriptorBufferInfo bufferInfo = {};
      bufferInfo.buffer = uniform_buffers_[i];
      bufferInfo.offset = 0;
      bufferInfo.range = sizeof(UniformBufferObject);

      VkDescriptorImageInfo imageInfo = {};
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.imageView = texture_image_view_;
      imageInfo.sampler = texture_sampler_;

      std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

      descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[0].dstSet = descriptor_sets_[i];
      descriptorWrites[0].dstBinding = 0;
      descriptorWrites[0].dstArrayElement = 0;
      descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptorWrites[0].descriptorCount = 1;
      descriptorWrites[0].pBufferInfo = &bufferInfo;

      descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[1].dstSet = descriptor_sets_[i];
      descriptorWrites[1].dstBinding = 1;
      descriptorWrites[1].dstArrayElement = 0;
      descriptorWrites[1].descriptorType =
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorWrites[1].descriptorCount = 1;
      descriptorWrites[1].pImageInfo = &imageInfo;

      vkUpdateDescriptorSets(device_.vk_device(),
                             static_cast<uint32_t>(descriptorWrites.size()),
                             descriptorWrites.data(), 0, nullptr);
    }
  }

  void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkBuffer& buffer,
                     VkDeviceMemory& bufferMemory)
  {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_.vk_device(), &bufferInfo, nullptr, &buffer) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device_.vk_device(), buffer,
                                  &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        find_memory_type(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device_.vk_device(), &allocInfo, nullptr,
                         &bufferMemory) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device_.vk_device(), buffer, bufferMemory, 0);
  }

  VkCommandBuffer begin_single_time_commands()
  {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = command_pool_;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device_.vk_device(), &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
  }

  void end_single_time_commands(VkCommandBuffer commandBuffer)
  {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(device_.graphics_queue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(device_.graphics_queue());

    vkFreeCommandBuffers(device_.vk_device(), command_pool_, 1, &commandBuffer);
  }

  void copy_buffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
  {
    VkCommandBuffer commandBuffer = begin_single_time_commands();

    VkBufferCopy copyRegion = {};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    end_single_time_commands(commandBuffer);
  }

  uint32_t find_memory_type(uint32_t typeFilter,
                            VkMemoryPropertyFlags properties)
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

    throw std::runtime_error("failed to find suitable memory type!");
  }

  void create_command_buffers()
  {
    command_buffers_.resize(swapChainFramebuffers.size());

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = command_pool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount =
        static_cast<uint32_t>(command_buffers_.size());

    if (vkAllocateCommandBuffers(device_.vk_device(), &allocInfo,
                                 command_buffers_.data()) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate command buffers!");
    }

    for (size_t i = 0; i < command_buffers_.size(); i++) {
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

      if (vkBeginCommandBuffer(command_buffers_[i], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
      }

      VkRenderPassBeginInfo renderPassInfo = {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = render_pass_;
      renderPassInfo.framebuffer = swapChainFramebuffers[i];
      renderPassInfo.renderArea.offset = {0, 0};
      renderPassInfo.renderArea.extent = swapChainExtent;

      std::array<VkClearValue, 2> clearValues = {};
      clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
      clearValues[1].depthStencil = {1.0f, 0};

      renderPassInfo.clearValueCount =
          static_cast<uint32_t>(clearValues.size());
      renderPassInfo.pClearValues = clearValues.data();

      vkCmdBeginRenderPass(command_buffers_[i], &renderPassInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      vkCmdBindPipeline(command_buffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphics_pipeline_);

      VkBuffer vertexBuffers[] = {vertex_buffer_};
      VkDeviceSize offsets[] = {0};
      vkCmdBindVertexBuffers(command_buffers_[i], 0, 1, vertexBuffers, offsets);

      vkCmdBindIndexBuffer(command_buffers_[i], index_buffer_, 0,
                           VK_INDEX_TYPE_UINT32);

      vkCmdBindDescriptorSets(command_buffers_[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                              0, 1, &descriptor_sets_[i], 0, nullptr);

      vkCmdDrawIndexed(command_buffers_[i],
                       static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);

      vkCmdEndRenderPass(command_buffers_[i]);

      if (vkEndCommandBuffer(command_buffers_[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }
  }

  void create_sync_objects()
  {
    image_available_semaphores_.resize(max_frames_in_flight);
    render_finished_semaphores_.resize(max_frames_in_flight);
    in_flight_fences_.resize(max_frames_in_flight);
    images_in_flight_.resize(swapchain_images_.size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < max_frames_in_flight; i++) {
      if (vkCreateSemaphore(device_.vk_device(), &semaphoreInfo, nullptr,
                            &image_available_semaphores_[i]) != VK_SUCCESS ||
          vkCreateSemaphore(device_.vk_device(), &semaphoreInfo, nullptr,
                            &render_finished_semaphores_[i]) != VK_SUCCESS ||
          vkCreateFence(device_.vk_device(), &fenceInfo, nullptr,
                        &in_flight_fences_[i]) != VK_SUCCESS) {
        throw std::runtime_error(
            "failed to create synchronization objects for a frame!");
      }
    }
  }

  void update_uniform_buffer(uint32_t currentImage)
  {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     currentTime - startTime)
                     .count();

    UniformBufferObject ubo = {};
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(10.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view =
        glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f),
                                static_cast<float>(swapChainExtent.width) /
                                    static_cast<float>(swapChainExtent.height),
                                0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    void* data;
    vkMapMemory(device_.vk_device(), uniform_buffers_memory_[currentImage], 0,
                sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device_.vk_device(), uniform_buffers_memory_[currentImage]);
  }

  void drawFrame()
  {
    vkWaitForFences(device_.vk_device(), 1, &in_flight_fences_[current_frame_],
                    VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result =
        vkAcquireNextImageKHR(device_.vk_device(), swapchain_, UINT64_MAX,
                              image_available_semaphores_[current_frame_],
                              VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      recreateSwapChain();
      return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      throw std::runtime_error("failed to acquire swap chain image!");
    }

    update_uniform_buffer(imageIndex);

    if (images_in_flight_[imageIndex] != VK_NULL_HANDLE) {
      vkWaitForFences(device_.vk_device(), 1, &images_in_flight_[imageIndex],
                      VK_TRUE, UINT64_MAX);
    }
    images_in_flight_[imageIndex] = in_flight_fences_[current_frame_];

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {
        image_available_semaphores_[current_frame_]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffers_[imageIndex];

    VkSemaphore signalSemaphores[] = {
        render_finished_semaphores_[current_frame_]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vkResetFences(device_.vk_device(), 1, &in_flight_fences_[current_frame_]);

    if (vkQueueSubmit(device_.graphics_queue(), 1, &submitInfo,
                      in_flight_fences_[current_frame_]) != VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {swapchain_};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(device_.present_queue(), &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
        framebuffer_resized) {
      framebuffer_resized = false;
      recreateSwapChain();
    } else if (result != VK_SUCCESS) {
      throw std::runtime_error("failed to present swap chain image!");
    }

    current_frame_ = (current_frame_ + 1) % max_frames_in_flight;
  }

  VkShaderModule createShaderModule(const std::vector<char>& code)
  {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device_.vk_device(), &createInfo, nullptr,
                             &shaderModule) != VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
  }

  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR>& availableFormats)
  {
    for (const auto& availableFormat : availableFormats) {
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return availableFormat;
      }
    }

    return availableFormats[0];
  }

  VkPresentModeKHR chooseSwapPresentMode(
      const std::vector<VkPresentModeKHR>& availablePresentModes)
  {
    for (const auto& availablePresentMode : availablePresentModes) {
      if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return availablePresentMode;
      }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
  }

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
  {
    if (capabilities.currentExtent.width != UINT32_MAX) {
      return capabilities.currentExtent;
    } else {
      int width, height;
      glfwGetFramebufferSize(window_.get(), &width, &height);

      VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                                 static_cast<uint32_t>(height)};

      actualExtent.width = std::max(
          capabilities.minImageExtent.width,
          std::min(capabilities.maxImageExtent.width, actualExtent.width));
      actualExtent.height = std::max(
          capabilities.minImageExtent.height,
          std::min(capabilities.maxImageExtent.height, actualExtent.height));

      return actualExtent;
    }
  }

  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
  {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, device_.surface(),
                                              &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, device_.surface(),
                                         &formatCount, nullptr);

    if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(
          device, device_.surface(), &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, device_.surface(),
                                              &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, device_.surface(),
                                                &presentModeCount,
                                                details.presentModes.data());
    }

    return details;
  }

  bool isDeviceSuitable(VkPhysicalDevice device)
  {
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
      swapChainAdequate = !swapChainSupport.formats.empty() &&
                          !swapChainSupport.presentModes.empty();
    }

    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return indices.isComplete() && extensionsSupported && swapChainAdequate &&
           supportedFeatures.samplerAnisotropy;
  }

  bool checkDeviceExtensionSupport(VkPhysicalDevice device)
  {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice pd)
  {
    QueueFamilyIndices ids;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &queueFamilyCount,
                                             queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        ids.graphicsFamily = i;
      }

      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(pd, static_cast<uint32_t>(i),
                                           device_.surface(), &presentSupport);

      if (presentSupport) {
        ids.presentFamily = i;
      }

      if (ids.isComplete()) {
        break;
      }

      i++;
    }

    return ids;
  }

  static std::vector<char> read_file(const std::string& filename)
  {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file " + filename + "!");
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

    return buffer;
  }
};

auto main() -> int
{
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
