#include <immintrin.h>
#include <string.h>
#include <math.h>
#include "processing.h"
#include "raytrace.h"

static float linearRGB_to_SRGB(const float value) {
  if (value <= 0.0031308f) {
    return 12.92f * value;
  }
  else {
    return 1.055f * powf(value, 0.416666666667f) - 0.055f;
  }
}

void frame_buffer_to_8bit_image(Camera camera, raytrace_instance* instance, RGB8* image) {
  RGBF* error_table = (RGBF*) malloc(sizeof(RGBF) * (instance->width + 2));

  memset(error_table, 0, sizeof(RGBF) * (instance->width + 2));

  for (int j = 0; j < instance->height; j++) {
    for (int i = 0; i < instance->width; i++) {
      RGB8 pixel;
      RGBF pixel_float = instance->frame_buffer[i + instance->width * j];

      RGBF color;
      color.r = min(255.9f, linearRGB_to_SRGB(pixel_float.r) * 255.9f + error_table[i + 1].r);
      color.g = min(255.9f, linearRGB_to_SRGB(pixel_float.g) * 255.9f + error_table[i + 1].g);
      color.b = min(255.9f, linearRGB_to_SRGB(pixel_float.b) * 255.9f + error_table[i + 1].b);

      error_table[i + 1].r = 0.0f;
      error_table[i + 1].g = 0.0f;
      error_table[i + 1].b = 0.0f;

      pixel.r = (uint8_t) color.r;
      pixel.g = (uint8_t) color.g;
      pixel.b = (uint8_t) color.b;

      RGBF error;
      error.r = 0.25f * (color.r - (float) pixel.r);
      error.g = 0.25f * (color.g - (float) pixel.g);
      error.b = 0.25f * (color.b - (float) pixel.b);

      error_table[i].r += error.r;
      error_table[i].g += error.g;
      error_table[i].b += error.b;

      error_table[i + 1].r += error.r;
      error_table[i + 1].g += error.g;
      error_table[i + 1].b += error.b;

      error_table[i + 2].r += 2.0f * error.r;
      error_table[i + 2].g += 2.0f * error.g;
      error_table[i + 2].b += 2.0f * error.b;

      image[i + instance->width * j] = pixel;
    }
  }

  free(error_table);
}

void frame_buffer_to_16bit_image(Camera camera, raytrace_instance* instance, RGB16* image) {
  RGBF* error_table = (RGBF*) malloc(sizeof(RGBF) * (instance->width + 2));

  memset(error_table, 0, sizeof(RGBF) * (instance->width + 2));

  for (int j = 0; j < instance->height; j++) {
    for (int i = 0; i < instance->width; i++) {
      RGB16 pixel;
      RGBF pixel_float = instance->frame_buffer[i + instance->width * j];

      RGBF color;
      color.r = min(65535.9f, linearRGB_to_SRGB(pixel_float.r) * 65535.9f + error_table[i + 1].r);
      color.g = min(65535.9f, linearRGB_to_SRGB(pixel_float.g) * 65535.9f + error_table[i + 1].g);
      color.b = min(65535.9f, linearRGB_to_SRGB(pixel_float.b) * 65535.9f + error_table[i + 1].b);

      error_table[i + 1].r = 0.0f;
      error_table[i + 1].g = 0.0f;
      error_table[i + 1].b = 0.0f;

      pixel.r = (uint16_t) color.r;
      pixel.g = (uint16_t) color.g;
      pixel.b = (uint16_t) color.b;

      RGBF error;
      error.r = 0.25f * (color.r - (float) pixel.r);
      error.g = 0.25f * (color.g - (float) pixel.g);
      error.b = 0.25f * (color.b - (float) pixel.b);

      error_table[i].r += error.r;
      error_table[i].g += error.g;
      error_table[i].b += error.b;

      error_table[i + 1].r += error.r;
      error_table[i + 1].g += error.g;
      error_table[i + 1].b += error.b;

      error_table[i + 2].r += 2.0f * error.r;
      error_table[i + 2].g += 2.0f * error.g;
      error_table[i + 2].b += 2.0f * error.b;

      image[i + instance->width * j] = pixel;
    }
  }

  free(error_table);

  int total    = instance->width * instance->height;
  __m128i mask = _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14, 15);

  int i;

  for (i = 0; i < total - 2; i += 2) {
    __m128i* image_offset = (__m128i*)(image + i);

    __m128i pixels = _mm_loadu_si128(image_offset);
    pixels         = _mm_shuffle_epi8(pixels, mask);
    _mm_storeu_si128(image_offset, pixels);
  }

  for (; i < total; i++) {
    RGB16 pixel  = image[i];
    uint8_t low  = pixel.r << 8;
    uint8_t high = pixel.r >> 8;
    pixel.r      = low + high;
    low          = pixel.g << 8;
    high         = pixel.g >> 8;
    pixel.g      = low + high;
    low          = pixel.b << 8;
    high         = pixel.b >> 8;
    pixel.b      = low + high;
    image[i]     = pixel;
  }
}
