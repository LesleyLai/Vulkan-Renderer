#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D albedoMapSampler;
layout(binding = 2) uniform sampler2D normalMapSampler;
layout(binding = 3) uniform sampler2D metallicMapSampler;
layout(binding = 4) uniform sampler2D roughnessMapSampler;

layout(location = 0) in vec3 WorldPos;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec2 TexCoord;
layout(location = 3) in vec3 CameraPos;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// point lights
const vec3 lightPositions[4] = {
vec3(-10.0f, 10.0f, 10.0f),
vec3(10.0f, 10.0f, 10.0f),
vec3(-10.0f, -10.0f, 10.0f),
vec3(10.0f, -10.0f, 10.0f),
};

// The radiant flux of point lights
const vec3 lightPower[4] = {
vec3(23.47, 21.31, 20.79),
vec3(23.47, 21.31, 20.79),
vec3(23.47, 21.31, 20.79),
vec3(23.47, 21.31, 20.79)
};

// See http://www.thetenthplanet.de/archives/1180
vec3 getNormalFromMap()
{
    //    vec3 tangentNormal = texture(normalMapSampler, TexCoord).xyz * 2.0 - 1.0;
    //
    //    vec3 Q1  = dFdx(WorldPos);
    //    vec3 Q2  = dFdy(WorldPos);
    //    vec2 st1 = dFdx(TexCoord);
    //    vec2 st2 = dFdy(TexCoord);
    //
    //    vec3 N   = normalize(Normal);
    //    vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
    //    vec3 B  = -normalize(cross(N, T));
    //    mat3 TBN = mat3(T, B, N);
    //
    //    return normalize(TBN * tangentNormal);

    return Normal;
}


float calculateAttenuation(vec3 WorldPos, vec3 lightPos) {
    vec3 D = lightPos - WorldPos;
    float distance_square = dot(D, D);
    return 1.0 / distance_square;
}


vec3 schlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}


void main() {

    vec3 albedo = pow(texture(albedoMapSampler, TexCoord).rgb, vec3(2.2));
    float metallic = texture(metallicMapSampler, TexCoord).r;
    float roughness = texture(roughnessMapSampler, TexCoord).r;
    const float ao = 0.01;

    vec3 N = getNormalFromMap();
    vec3 V = normalize(CameraPos - WorldPos);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 4; ++i) {

        vec3 L = normalize(lightPositions[i] - WorldPos);
        vec3 H = normalize(V + L);

        float attenuation = calculateAttenuation(WorldPos, lightPositions[i]);
        vec3 radiance = lightPower[i] * attenuation;

        // Cook-Torrance BRDF
        // {

        float NDF = DistributionGGX(N, H, roughness);
        float G   = GeometrySmith(N, V, L, roughness);
        vec3 F  = schlick(max(dot(H, V), 0.0), F0);

        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;// 0.001 to prevent divide by zero.;
        vec3 specular     = numerator / max(denominator, 0.001);
        // }

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;

        kD *= 1.0 - metallic;

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }


    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color   = ambient + Lo;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    outColor = vec4(color, 1.0);
}