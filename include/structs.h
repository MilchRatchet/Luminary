#ifndef STRUCTS_H
#define STRUCTS_H

#include <cuda_runtime_api.h>
#include <stdint.h>

// This struct is stored as a struct of arrays, members are grouped into 16 bytes where possible. Padding is not required.
#define INTERLEAVED_STORAGE

/********************************************************
 * Vectors and matrices
 ********************************************************/

struct vec3 {
  float x;
  float y;
  float z;
} typedef vec3;

struct vec4 {
  float x;
  float y;
  float z;
  float w;
} typedef vec4;

struct Quaternion {
  float w;
  float x;
  float y;
  float z;
} typedef Quaternion;

struct Mat4x4 {
  float f11;
  float f12;
  float f13;
  float f14;
  float f21;
  float f22;
  float f23;
  float f24;
  float f31;
  float f32;
  float f33;
  float f34;
  float f41;
  float f42;
  float f43;
  float f44;
} typedef Mat4x4;

struct Mat3x3 {
  float f11;
  float f12;
  float f13;
  float f21;
  float f22;
  float f23;
  float f31;
  float f32;
  float f33;
} typedef Mat3x3;

struct UV {
  float u;
  float v;
} typedef UV;

/********************************************************
 * Mesh
 ********************************************************/

INTERLEAVED_STORAGE struct Triangle {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  vec3 vertex_normal;
  vec3 edge1_normal;
  vec3 edge2_normal;
  UV vertex_texture;
  UV edge1_texture;
  UV edge2_texture;
  uint32_t material_id;
  uint32_t light_id;
} typedef Triangle;

struct TraversalTriangle {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  uint32_t albedo_tex;
  uint32_t id;
  float padding2;
} typedef TraversalTriangle;

struct TriangleLight {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  uint32_t triangle_id;
  uint32_t material_id;
  float padding2;
} typedef TriangleLight;

// Traditional description through vertex buffer and index buffer which is required for OptiX RT.
// Both vertex buffer and index buffer have a stride of 16 bytes for each triplet
// but the counts only indicate the number of actual data entries.
struct TriangleGeomData {
  float* vertex_buffer;
  uint32_t* index_buffer;
  uint32_t vertex_count;
  uint32_t index_count;
  uint32_t triangle_count;
} typedef TriangleGeomData;

struct Node8 {
  vec3 p;
  int8_t ex;
  int8_t ey;
  int8_t ez;
  uint8_t imask;
  int32_t child_node_base_index;
  int32_t triangle_base_index;
  uint8_t meta[8];
  uint8_t low_x[8];
  uint8_t low_y[8];
  uint8_t low_z[8];
  uint8_t high_x[8];
  uint8_t high_y[8];
  uint8_t high_z[8];
} typedef Node8;

struct Quad {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  vec3 normal;
} typedef Quad;

/********************************************************
 * Pixelformats
 ********************************************************/

struct RGB8 {
  uint8_t r;
  uint8_t g;
  uint8_t b;
} typedef RGB8;

struct RGBA8 {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
} typedef RGBA8;

struct XRGB8 {
  uint8_t b;
  uint8_t g;
  uint8_t r;
  uint8_t ignore;
} typedef XRGB8;

struct RGB16 {
  uint16_t r;
  uint16_t g;
  uint16_t b;
} typedef RGB16;

struct RGBA16 {
  uint16_t r;
  uint16_t g;
  uint16_t b;
  uint16_t a;
} typedef RGBA16;

struct RGBF {
  float r;
  float g;
  float b;
} typedef RGBF;

#ifndef __cplusplus
struct RGBAhalf {
  uint16_t r;
  uint16_t g;
  uint16_t b;
  uint16_t a;
} typedef RGBAhalf;
#else
#include <cuda_fp16.h>

struct RGBAhalf {
  __half2 rg;
  __half2 ba;
} typedef RGBAhalf;
#endif

struct RGBAF {
  float r;
  float g;
  float b;
  float a;
} typedef RGBAF;

/********************************************************
 * Textures
 ********************************************************/

struct PackedMaterial {
  float refraction_index;
  uint16_t albedo_r;
  uint16_t albedo_g;
  uint16_t albedo_b;
  uint16_t albedo_a;
  uint16_t emission_r;
  uint16_t emission_g;
  uint16_t emission_b;
  uint16_t emission_scale;
  uint16_t metallic;
  uint16_t roughness;
  uint16_t albedo_map;
  uint16_t luminance_map;
  uint16_t material_map;
  uint16_t normal_map;
} typedef PackedMaterial;

struct Material {
  float refraction_index;
  RGBAF albedo;
  RGBF emission;
  float metallic;
  float roughness;
  uint16_t albedo_map;
  uint16_t luminance_map;
  uint16_t material_map;
  uint16_t normal_map;
} typedef Material;

struct TextureG {
  unsigned int width;
  unsigned int height;
  float* data;
} typedef TextureG;

enum TextureDataType { TexDataFP32 = 0, TexDataUINT8 = 1, TexDataUINT16 = 2 } typedef TextureDataType;
enum TextureWrappingMode { TexModeWrap = 0, TexModeClamp = 1, TexModeMirror = 2, TexModeBorder = 3 } typedef TextureWrappingMode;
enum TextureDimensionType { Tex2D = 0, Tex3D = 1 } typedef TextureDimensionType;
enum TextureStorageLocation { TexStorageCPU = 0, TexStorageGPU = 1 } typedef TextureStorageLocation;
enum TextureFilterMode { TexFilterPoint = 0, TexFilterLinear = 1 } typedef TextureFilterMode;
enum TextureMipmapMode { TexMipmapNone = 0, TexMipmapGenerate = 1 } typedef TextureMipmapMode;
enum TextureReadMode { TexReadModeNormalized = 0, TexReadModeElement = 1 } typedef TextureReadMode;

struct TextureRGBA {
  unsigned int width;
  unsigned int height;
  unsigned int depth;
  unsigned int pitch;
  TextureDataType type;
  TextureWrappingMode wrap_mode_S;
  TextureWrappingMode wrap_mode_T;
  TextureWrappingMode wrap_mode_R;
  TextureDimensionType dim;
  TextureStorageLocation storage;
  TextureFilterMode filter;
  TextureMipmapMode mipmap;
  TextureReadMode read_mode;
  int mipmap_max_level;
  void* data;
  float gamma;
  unsigned int num_components;
} typedef TextureRGBA;

struct DeviceTexture {
  cudaTextureObject_t tex;
  float inv_width;
  float inv_height;
  float gamma;
} typedef DeviceTexture;

struct LightSample {
  uint32_t seed;
  uint32_t presampled_id;
  uint32_t id;
  float weight;
} typedef LightSample;

enum GBufferFlags {
  G_BUFFER_REQUIRES_SAMPLING    = 0b1,
  G_BUFFER_VOLUME_HIT           = 0b10,
  G_BUFFER_REFRACTION_IS_INSIDE = 0b100
} typedef GBufferFlags;

struct GBufferData {
  uint32_t hit_id;
  RGBAF albedo;
  RGBF emission;
  vec3 position;
  vec3 V;
  vec3 normal;
  float roughness;
  float metallic;
  uint32_t flags;
  float ior_in;
  float ior_out;
} typedef GBufferData;

////////////////////////////////////////////////////////////////////
// Kernel passing structs
////////////////////////////////////////////////////////////////////

//
// Shading task structs. They used to be very different, however, they could all be unified as of October 2023.
//
struct GeometryTask {
  ushort2 index;
  vec3 position;
  vec3 ray;
  uint32_t hit_id;
} typedef GeometryTask;

struct ParticleTask {
  ushort2 index;
  vec3 position;
  vec3 ray;
  uint32_t hit_id;
} typedef ParticleTask;

struct SkyTask {
  ushort2 index;
  vec3 origin;
  vec3 ray;
  uint32_t padding;
} typedef SkyTask;

struct OceanTask {
  ushort2 index;
  vec3 position;
  vec3 ray;
  float padding;
} typedef OceanTask;

struct ToyTask {
  ushort2 index;
  vec3 position;
  vec3 ray;
  uint32_t padding;
} typedef ToyTask;

struct VolumeTask {
  ushort2 index;
  vec3 position;
  vec3 ray;
  uint32_t hit_id;
} typedef VolumeTask;

struct TraceTask {
  vec3 origin;
  vec3 ray;
  ushort2 index;
  uint32_t padding;
} typedef TraceTask;

struct TraceResult {
  float depth;
  uint32_t hit_id;
} typedef TraceResult;

#endif /* STRUCTS_H */
