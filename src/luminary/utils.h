#ifndef LUMINARY_UTILS_H
#define LUMINARY_UTILS_H

#include "internal_api_resolve.h"

// API definitions must first be translated

#include <assert.h>

#include "internal_array.h"
#include "sky_defines.h"

#ifndef PI
#define PI 3.141592653589f
#endif

// https://stackoverflow.com/a/9283155
#define LITTLE_ENDIAN 0x41424344UL
#define BIG_ENDIAN 0x44434241UL
#define ENDIAN_ORDER ('ABCD')

#define LUM_STATIC_SIZE_ASSERT(struct, size) static_assert(sizeof(struct) == size, #struct " has invalid size")
#define LUM_ASSUME(assumption) static_assert((assumption), "Assumption: " #assumption " was violated.");

// Flags variables as unused so that no warning is emitted
#define LUM_UNUSED(x) ((void) (x))

// This struct is stored as a struct of arrays, members are grouped into 16 bytes where possible. Padding is not required.
#define INTERLEAVED_STORAGE
// Round the size of an element in an interleaved buffer to the next multiple of 16 bytes
#define INTERLEAVED_ALLOCATION_SIZE(X) ((X + 15u) & (~0xFu))

#define TEXTURE_NONE ((uint16_t) 0xffffu)

#define MATERIAL_ID_INVALID 0xFFFF
#define INSTANCE_ID_INVALID 0xFFFF
#define DEPTH_INVALID -1.0f

#define LIGHT_ID_SUN (0xffffffffu)
#define LIGHT_ID_NONE (0xfffffff1u)
#define LIGHT_ID_ANY (0xfffffff0u)
#define LIGHT_ID_TRIANGLE_ID_LIMIT (0x7fffffffu)

// Print stats for the work queues
#define LUMINARY_WORK_QUEUE_STATS_PRINT
#define LUMINARY_WORK_QUEUE_STATS_PRINT_THRESHOLD 0.01

enum VolumeType { VOLUME_TYPE_FOG = 0, VOLUME_TYPE_OCEAN = 1, VOLUME_TYPE_PARTICLE = 2, VOLUME_TYPE_NONE = 0xFFFFFFFF } typedef VolumeType;

typedef LuminaryResult (*QueueEntryFunction)(void* worker, void* args);

struct QueueEntry {
  const char* name;
  LuminaryResult (*function)(void* worker, void* args);
  LuminaryResult (*clear_func)(void* worker, void* args);
  void* args;
  bool remove_duplicates;
  bool queuer_cannot_execute;  // Used to avoid self execution on CUDA callback threads.
} typedef QueueEntry;

struct Quaternion {
  float x;
  float y;
  float z;
  float w;
} typedef Quaternion;

struct Quaternion16 {
  uint16_t x;
  uint16_t y;
  uint16_t z;
  uint16_t w;
} typedef Quaternion16;

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

////////////////////////////////////////////////////////////////////
// Mesh
////////////////////////////////////////////////////////////////////

struct Triangle {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  vec3 vertex_normal;
  vec3 edge1_normal;
  vec3 edge2_normal;
  UV vertex_texture;
  UV edge1_texture;
  UV edge2_texture;
  uint16_t material_id;
} typedef Triangle;

struct TraversalTriangle {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  uint32_t albedo_tex;
  uint32_t id;
  float padding2;
} typedef TraversalTriangle;

struct Star {
  float altitude;
  float azimuth;
  float radius;
  float intensity;
} typedef Star;
LUM_STATIC_SIZE_ASSERT(Star, 0x10u);

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

struct OutputDescriptor {
  bool is_recurring_output;
  struct {
    uint32_t width;
    uint32_t height;
    uint32_t sample_count;
    bool is_first_output;
    float time;
  } meta_data;
  void* data;
} typedef OutputDescriptor;

#endif /* LUMINARY_UTILS_H */
