#ifndef LUMINARY_UTILS_H
#define LUMINARY_UTILS_H

#include "internal_api_resolve.h"

// API definitions must first be translated

#include <assert.h>

#include "sky_defines.h"

typedef LuminaryResult (*QueueEntryFunction)(void* worker, void* args);

struct QueueEntry {
  const char* name;
  LuminaryResult (*function)(void* worker, void* args);
  void* args;
} typedef QueueEntry;

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

#ifndef PI
#define PI 3.141592653589f
#endif

#define LUM_STATIC_SIZE_ASSERT(struct, size) static_assert(sizeof(struct) == size, #struct " has invalid size")

// Flags variables as unused so that no warning is emitted
#define LUM_UNUSED(x) ((void) (x))

#endif /* LUMINARY_UTILS_H */
