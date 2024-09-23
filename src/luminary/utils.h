#ifndef LUMINARY_UTILS_H
#define LUMINARY_UTILS_H

#include "internal_api_resolve.h"

// API definitions must first be translated

#include <assert.h>

#include "sky_defines.h"

// This struct is stored as a struct of arrays, members are grouped into 16 bytes where possible. Padding is not required.
#define INTERLEAVED_STORAGE
// Round the size of an element in an interleaved buffer to the next multiple of 16 bytes
#define INTERLEAVED_ALLOCATION_SIZE(X) ((X + 15u) & (~0xFu))

#define TEXTURE_NONE ((uint16_t) 0xffffu)

enum LightID {
  LIGHT_ID_SUN               = 0xffffffffu,
  LIGHT_ID_TOY               = 0xfffffffeu,
  LIGHT_ID_NONE              = 0xfffffff1u,
  LIGHT_ID_ANY               = 0xfffffff0u,
  LIGHT_ID_TRIANGLE_ID_LIMIT = 0x7fffffffu
} typedef LightID;

typedef LuminaryResult (*QueueEntryFunction)(void* worker, void* args);

struct QueueEntry {
  const char* name;
  LuminaryResult (*function)(void* worker, void* args);
  void* args;
} typedef QueueEntry;

struct Quaternion {
  float w;
  float x;
  float y;
  float z;
} typedef Quaternion;

struct UV {
  float u;
  float v;
} typedef UV;

////////////////////////////////////////////////////////////////////
// Dirty Flags
////////////////////////////////////////////////////////////////////

typedef uint64_t DirtyFlags;

enum DirtyFlagsDefinitions {
  DIRTY_FLAG_OUTPUT        = 0x0000000000000001ull,
  DIRTY_FLAG_SAMPLES       = 0x0000000000000002ull,
  DIRTY_FLAG_IMAGE_BUFFERS = 0x0000000000000004ull
};

////////////////////////////////////////////////////////////////////
// Pixelformats
////////////////////////////////////////////////////////////////////

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

struct RGBAF {
  float r;
  float g;
  float b;
  float a;
} typedef RGBAF;

////////////////////////////////////////////////////////////////////
// Material
////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////
// Mesh
////////////////////////////////////////////////////////////////////

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
  float power;
} typedef TriangleLight;
static_assert(sizeof(TriangleLight) == 0x30, "Incorrect packing size.");

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

struct BVHNode8 {
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
} typedef BVHNode8;

#ifndef PI
#define PI 3.141592653589f
#endif

#define LUM_STATIC_SIZE_ASSERT(struct, size) static_assert(sizeof(struct) == size, #struct " has invalid size")

// Flags variables as unused so that no warning is emitted
#define LUM_UNUSED(x) ((void) (x))

#endif /* LUMINARY_UTILS_H */
