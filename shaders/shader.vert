#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 cameraPos;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 WorldPos;
layout(location = 1) out vec3 Normal;
layout(location = 2) out vec2 TexCoord;
layout(location = 3) out vec3 CameraPos;

void main() {
    WorldPos = vec3(ubo.model * vec4(inPosition, 1.0));
    Normal = mat3(ubo.model) * inNormal;
    TexCoord = inTexCoord;
    CameraPos = ubo.cameraPos;

    gl_Position = ubo.proj * ubo.view * vec4(WorldPos, 1.0);
}

