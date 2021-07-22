#include <immintrin.h>
#include <string.h>
#include <math.h>
#include "processing.h"
#include "raytrace.h"
#include "error.h"

static float linearRGB_to_SRGB(const float value) {
  if (value <= 0.0031308f) {
    return 12.92f * value;
  }
  else {
    return 1.055f * powf(value, 0.416666666667f) - 0.055f;
  }
}

void frame_buffer_to_8bit_image(Camera camera, RaytraceInstance* instance, RGB8* image) {
  RGBF* error_table = (RGBF*) malloc(sizeof(RGBF) * (instance->width + 2));

  assert((unsigned long long) error_table, "Failed to allocate memory!", 1);

  memset(error_table, 0, sizeof(RGBF) * (instance->width + 2));

  for (int j = 0; j < instance->height; j++) {
    for (int i = 0; i < instance->width; i++) {
      RGB8 pixel;
      RGBF pixel_float = instance->frame_output[i + instance->width * j];

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

void frame_buffer_to_16bit_image(Camera camera, RaytraceInstance* instance, RGB16* image) {
  RGBF* error_table = (RGBF*) malloc(sizeof(RGBF) * (instance->width + 2));

  assert((unsigned long long) error_table, "Failed to allocate memory!", 1);

  memset(error_table, 0, sizeof(RGBF) * (instance->width + 2));

  for (int j = 0; j < instance->height; j++) {
    for (int i = 0; i < instance->width; i++) {
      RGB16 pixel;
      RGBF pixel_float = instance->frame_output[i + instance->width * j];

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
    __m128i* image_offset = (__m128i*) (image + i);

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

void post_bloom(RaytraceInstance* instance, const float sigma, const float strength) {
  unsigned int pixel_count = instance->width * instance->height;

  RGBF* image = instance->frame_output;

  RGBF* illuminance = (RGBF*) _mm_malloc(sizeof(RGBF) * pixel_count, 32);

  unsigned int i = 0;

  __m256 ones = _mm256_set1_ps(1.0f);

  float* image_ptr       = (float*) image;
  float* illuminance_ptr = (float*) illuminance;

  for (; i + 8 < pixel_count * 3; i += 8) {
    __m256 pixels = _mm256_load_ps(image_ptr);

    __m256 truncated_pixels = _mm256_min_ps(pixels, ones);

    _mm256_store_ps(image_ptr, truncated_pixels);

    __m256 excess = _mm256_sub_ps(pixels, truncated_pixels);

    _mm256_store_ps(illuminance_ptr, excess);

    image_ptr += 8;
    illuminance_ptr += 8;
  }

  for (; i < pixel_count * 3; i++) {
    const float val = *image_ptr;

    const float truncated_val = (val > 1.0f) ? 1.0f : val;

    *image_ptr = truncated_val;

    const float excess = val - truncated_val;

    *illuminance_ptr = excess;

    image_ptr++;
    illuminance_ptr++;
  }

  RGBF* temp = (RGBF*) _mm_malloc(sizeof(RGBF) * pixel_count, 32);

  illuminance_ptr = (float*) illuminance;
  float* temp_ptr = (float*) temp;

  memset(temp_ptr, 0.0f, sizeof(float) * pixel_count * 3);

  const float reference_pixel_count = 1280.0f;

  const float pixel_size = reference_pixel_count / instance->width;

  const int kernel_size = (int) ((sigma * 3.0f) / pixel_size) + 1;

  float* gauss_kernel = (float*) _mm_malloc(sizeof(float) * kernel_size, 32);

  const float gauss_factor1 = 1.0f / (2.0f * sigma * sigma);
  const float gauss_factor2 = gauss_factor1 / 3.14159265358979323846f;

  for (int i = 0; i < kernel_size; i++) {
    const float x   = i * pixel_size;
    gauss_kernel[i] = strength * pixel_size * gauss_factor2 * expf(-x * x * gauss_factor1);
  }

  for (unsigned int i = 0; i < instance->height; i++) {
    for (unsigned int j = 0; j < instance->width * 3; j++) {
      const float val = *illuminance_ptr;

      if (val != 0.0f) {
        for (unsigned int k = 0; k < kernel_size && 3 * k + j < instance->width * 3; k++) {
          temp_ptr[i * instance->width * 3 + j + 3 * k] += gauss_kernel[k] * val;
        }

        for (unsigned int k = 1; k < kernel_size && j >= 3 * k; k++) {
          temp_ptr[i * instance->width * 3 + j - 3 * k] += gauss_kernel[k] * val;
        }
      }

      illuminance_ptr++;
    }
  }

  temp_ptr        = (float*) temp;
  illuminance_ptr = (float*) illuminance;

  for (unsigned int i = 0; i < instance->height; i++) {
    for (unsigned int j = 0; j < instance->width * 3; j++) {
      const float val = *temp_ptr;

      if (val != 0.0f) {
        for (unsigned int k = 0; k < kernel_size && i + k < instance->height; k++) {
          illuminance_ptr[(i + k) * instance->width * 3 + j] += gauss_kernel[k] * val;
        }

        for (unsigned int k = 1; k < kernel_size && i >= k; k++) {
          illuminance_ptr[(i - k) * instance->width * 3 + j] += gauss_kernel[k] * val;
        }
      }

      temp_ptr++;
    }
  }

  image_ptr = (float*) image;

  i = 0;

  for (; i + 8 < pixel_count * 3; i += 8) {
    __m256 pixels = _mm256_load_ps(image_ptr);

    __m256 blurred_illuminance = _mm256_load_ps(illuminance_ptr);

    pixels = _mm256_add_ps(pixels, blurred_illuminance);

    _mm256_store_ps(image_ptr, pixels);

    image_ptr += 8;
    illuminance_ptr += 8;
  }

  for (; i < pixel_count * 3; i++) {
    const float val = *image_ptr;

    const float blurred_illuminance = *illuminance_ptr;

    *image_ptr = val + blurred_illuminance;

    image_ptr++;
    illuminance_ptr++;
  }

  _mm_free(gauss_kernel);
  _mm_free(illuminance);
  _mm_free(temp);
}

static RGBF tonemap(RGBF pixel) {
  const float a = 2.51f;
  const float b = 0.03f;
  const float c = 2.43f;
  const float d = 0.59f;
  const float e = 0.14f;

  pixel.r = 1.25f * (pixel.r * (a * pixel.r + b)) / (pixel.r * (c * pixel.r + d) + e);
  pixel.g = 1.25f * (pixel.g * (a * pixel.g + b)) / (pixel.g * (c * pixel.g + d) + e);
  pixel.b = 1.25f * (pixel.b * (a * pixel.b + b)) / (pixel.b * (c * pixel.b + d) + e);

  return pixel;
}

void post_tonemapping(RaytraceInstance* instance) {
  const unsigned int pixel_count = instance->width * instance->height;

  RGBF* pixels = instance->frame_output;

  for (unsigned int i = 0; i < pixel_count; i++) {
    pixels[i] = tonemap(pixels[i]);
  }
}

static float color_distance(const RGBF a, const RGBF b) {
  const float x = a.r - b.r;
  const float y = a.g - b.g;
  const float z = a.b - b.b;
  return sqrtf(x * x + y * y + z * z);
}

void post_median_filter(RaytraceInstance* instance, const float bias) {
  const unsigned int pixel_count = instance->width * instance->height;
  RGBF* pixels                   = instance->frame_output;
  RGBF* new_pixels               = (RGBF*) _mm_malloc(sizeof(RGBF) * pixel_count, 32);

  RGBF* window = (RGBF*) malloc(sizeof(RGBF) * 9);

  float* distances = (float*) malloc(sizeof(float) * 9);

  memcpy(new_pixels, pixels, sizeof(RGBF) * pixel_count);

  for (unsigned int i = 1; i < instance->height - 1; i++) {
    for (unsigned int j = 1; j < instance->width - 1; j++) {
      memcpy(window, pixels + j - 1 + (i - 1) * instance->width, sizeof(RGBF) * 3);
      memcpy(window + 3, pixels + j - 1 + i * instance->width, sizeof(RGBF) * 3);
      memcpy(window + 6, pixels + j - 1 + (i + 1) * instance->width, sizeof(RGBF) * 3);
      memset(distances, 0.0f, sizeof(float) * 9);

      for (int k = 0; k < 9; k++) {
        float distance   = distances[k];
        const RGBF pixel = window[k];
        for (int l = k + 1; l < 9; l++) {
          if (l == 4)
            continue;
          const float res = color_distance(window[l], pixel);
          distance += res;
          distances[l] += res;
        }
        distances[k] = distance;
      }

      int min        = 4;
      float min_dist = distances[4] * bias;

      for (int k = 0; k < 9; k++) {
        if (distances[k] < min_dist) {
          min_dist = distances[k];
          min      = k;
        }
      }

      new_pixels[j + i * instance->width] = window[min];
    }
  }

  free(window);
  free(distances);

  instance->frame_output = new_pixels;

  _mm_free(pixels);
}
