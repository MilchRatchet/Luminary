#include "image.h"

#include <ctype.h>
#include <luminary/image.h>
#include <stdlib.h>
#include <string.h>

#include "host_local_memory.h"
#include "internal_error.h"

static void* _image_malloc_stbi(size_t size) {
  void* data;
  LuminaryResult result = host_malloc(&data, size);

  return (result == LUMINARY_SUCCESS) ? data : (void*) 0;
}

static void _image_free_stbi(void* data) {
  if (data == (void*) 0)
    return;

  (void) host_free(&data);
}

static void* _image_realloc_stbi(void* data, size_t size) {
  LuminaryResult result;
  if (data == (void*) 0) {
    result = host_malloc(&data, size);
  }
  else {
    result = host_realloc(&data, size);
  }

  return (result == LUMINARY_SUCCESS) ? data : (void*) 0;
}

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_NO_FAILURE_STRINGS
#define STBI_ASSERT(x)
#ifdef __WIN32__
#define STBI_WINDOWS_UTF8
#endif /* __WIN32__ */

#define STBI_MALLOC(sz) _image_malloc_stbi(sz)
#define STBI_REALLOC(p, newsz) _image_realloc_stbi(p, newsz)
#define STBI_FREE(p) _image_free_stbi(p)

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STBIW_MALLOC(sz) _image_malloc_stbi(sz)
#define STBIW_REALLOC(p, newsz) _image_realloc_stbi(p, newsz)
#define STBIW_FREE(p) _image_free_stbi(p)

// Disable warnings about unused static functions in stb_image. We assume that all non-MSVC compilers support GCC style pragmas.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4505)
#else /* _MSC_VER && !__clang__ */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif /* !_MSC_VER || __clang__ */

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#else /* _MSC_VER && !__clang__ */
#pragma GCC diagnostic pop
#endif /* !_MSC_VER || __clang__ */

static LuminaryResult _image_load_hdr(Texture* texture, const uint8_t* file_mem, size_t file_length, const char* file_name) {
  __CHECK_NULL_ARGUMENT(texture);
  __CHECK_NULL_ARGUMENT(file_mem);

  int width, height, num_components;
  float* data = stbi_loadf_from_memory(file_mem, (int) file_length, &width, &height, &num_components, 4);

  if (data == (float*) 0) {
    __FAILURE_HANDLE(texture_invalidate(texture));
    error_message("Failed to load image %s.", file_name);
    return LUMINARY_SUCCESS;
  }

  __FAILURE_HANDLE(texture_fill(texture, width, height, 1, data, TEXTURE_DATA_TYPE_FP32, 4));

  // stb_image does not expose this so we have to guess
  texture->gamma = 1.0f;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _image_load_16(Texture* texture, const uint8_t* file_mem, size_t file_length, const char* file_name) {
  __CHECK_NULL_ARGUMENT(texture);
  __CHECK_NULL_ARGUMENT(file_mem);

  int width, height, num_components;
  uint16_t* data = stbi_load_16_from_memory(file_mem, (int) file_length, &width, &height, &num_components, 4);

  if (data == (uint16_t*) 0) {
    __FAILURE_HANDLE(texture_invalidate(texture));
    error_message("Failed to load image %s.", file_name);
    return LUMINARY_SUCCESS;
  }

  __FAILURE_HANDLE(texture_fill(texture, width, height, 1, data, TEXTURE_DATA_TYPE_U16, 4));

  // stb_image does not expose this so we have to guess
  texture->gamma = 2.2f;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _image_load_8(Texture* texture, const uint8_t* file_mem, size_t file_length, const char* file_name) {
  __CHECK_NULL_ARGUMENT(texture);
  __CHECK_NULL_ARGUMENT(file_mem);

  int width, height, num_components;
  uint8_t* data = stbi_load_from_memory(file_mem, (int) file_length, &width, &height, &num_components, 4);

  if (data == (uint8_t*) 0) {
    __FAILURE_HANDLE(texture_invalidate(texture));
    error_message("Failed to load image %s.", file_name);
    return LUMINARY_SUCCESS;
  }

  __FAILURE_HANDLE(texture_fill(texture, width, height, 1, data, TEXTURE_DATA_TYPE_U8, 4));

  // stb_image does not expose this so we have to guess
  // TODO: Expose this as a per texture setting through the API one day so users can manually fix this.
  texture->gamma = 2.2f;

  return LUMINARY_SUCCESS;
}

static bool _image_is_cube_from_memory(const uint8_t* file_mem, size_t file_length) {
  __CHECK_NULL_ARGUMENT(file_mem);

  const char* str        = (const char*) file_mem;
  const size_t check_len = file_length > 256 ? 256 : file_length;

  for (size_t i = 0; i < check_len; ++i) {
    if (i + 11 <= check_len && strncmp(str + i, "LUT_3D_SIZE", 11) == 0)
      return true;
    if (i + 11 <= check_len && strncmp(str + i, "LUT_1D_SIZE", 11) == 0)
      return true;
    if (i + 5 <= check_len && strncmp(str + i, "TITLE", 5) == 0)
      return true;
  }

  return false;
}

static void _fill_identity_cube(float* data, uint32_t size, uint32_t dim) {
  if (dim == 1) {
    for (uint32_t i = 0; i < size; ++i) {
      float v         = i / (float) (size > 1 ? (size - 1) : 1);
      data[i * 4 + 0] = v;
      data[i * 4 + 1] = v;
      data[i * 4 + 2] = v;
      data[i * 4 + 3] = 1.0f;
    }
  }
  else {
    for (uint32_t z = 0; z < size; ++z) {
      for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
          uint32_t idx  = (z * size * size + y * size + x) * 4;
          data[idx + 0] = x / (float) (size > 1 ? (size - 1) : 1);
          data[idx + 1] = y / (float) (size > 1 ? (size - 1) : 1);
          data[idx + 2] = z / (float) (size > 1 ? (size - 1) : 1);
          data[idx + 3] = 1.0f;
        }
      }
    }
  }
}

static LuminaryResult _image_load_cube(Texture* texture, const uint8_t* file_mem, size_t file_length, const char* file_name) {
  __CHECK_NULL_ARGUMENT(texture);
  __CHECK_NULL_ARGUMENT(file_mem);

  uint32_t size       = 0;
  uint32_t dim        = 3;
  float domain_min[3] = {0.0f, 0.0f, 0.0f};
  float domain_max[3] = {1.0f, 1.0f, 1.0f};
  bool has_domain_min = false;
  bool has_domain_max = false;

  const char* ptr = (const char*) file_mem;
  const char* end = ptr + file_length;

  while (ptr < end) {
    while (ptr < end && isspace((unsigned char) *ptr))
      ptr++;
    if (ptr >= end)
      break;

    if (*ptr == '#') {
      while (ptr < end && *ptr != '\n')
        ptr++;
      continue;
    }

    if (end - ptr >= 11 && strncmp(ptr, "LUT_3D_SIZE", 11) == 0) {
      ptr += 11;
      char* endptr;
      size = strtoul(ptr, &endptr, 10);
      ptr  = endptr;
    }
    else if (end - ptr >= 11 && strncmp(ptr, "LUT_1D_SIZE", 11) == 0) {
      ptr += 11;
      char* endptr;
      size = strtoul(ptr, &endptr, 10);
      dim  = 1;
      ptr  = endptr;
    }
    else if (end - ptr >= 10 && strncmp(ptr, "DOMAIN_MIN", 10) == 0) {
      ptr += 10;
      char* endptr;
      domain_min[0]  = strtof(ptr, &endptr);
      ptr            = endptr;
      domain_min[1]  = strtof(ptr, &endptr);
      ptr            = endptr;
      domain_min[2]  = strtof(ptr, &endptr);
      ptr            = endptr;
      has_domain_min = true;
    }
    else if (end - ptr >= 10 && strncmp(ptr, "DOMAIN_MAX", 10) == 0) {
      ptr += 10;
      char* endptr;
      domain_max[0]  = strtof(ptr, &endptr);
      ptr            = endptr;
      domain_max[1]  = strtof(ptr, &endptr);
      ptr            = endptr;
      domain_max[2]  = strtof(ptr, &endptr);
      ptr            = endptr;
      has_domain_max = true;
    }
    else if (end - ptr >= 5 && strncmp(ptr, "TITLE", 5) == 0) {
      ptr += 5;
      while (ptr < end && *ptr != '\n')
        ptr++;
    }
    else {
      break;
    }
  }

  // Handle malformed if size is 0
  if (size == 0)
    size = 2;  // Identity fallback size

  uint32_t total_elements = dim == 1 ? size : (size * size * size);

  float* data;
  __FAILURE_HANDLE(host_malloc(&data, total_elements * 4 * sizeof(float)));

  uint32_t i     = 0;
  bool malformed = false;
  while (ptr < end && i < total_elements) {
    while (ptr < end && isspace((unsigned char) *ptr))
      ptr++;
    if (ptr >= end)
      break;
    if (*ptr == '#') {
      while (ptr < end && *ptr != '\n')
        ptr++;
      continue;
    }

    if (i < total_elements) {
      char* endptr;
      float r = strtof(ptr, &endptr);
      if (ptr == endptr) {
        malformed = true;
        break;
      }
      ptr     = endptr;
      float g = strtof(ptr, &endptr);
      if (ptr == endptr) {
        malformed = true;
        break;
      }
      ptr     = endptr;
      float b = strtof(ptr, &endptr);
      if (ptr == endptr) {
        malformed = true;
        break;
      }
      ptr = endptr;

      if (has_domain_min && has_domain_max) {
        float range[3] = {domain_max[0] - domain_min[0], domain_max[1] - domain_min[1], domain_max[2] - domain_min[2]};
        r              = (r - domain_min[0]) / (range[0] != 0.0f ? range[0] : 1.0f);
        g              = (g - domain_min[1]) / (range[1] != 0.0f ? range[1] : 1.0f);
        b              = (b - domain_min[2]) / (range[2] != 0.0f ? range[2] : 1.0f);
      }

      data[i * 4 + 0] = r;
      data[i * 4 + 1] = g;
      data[i * 4 + 2] = b;
      data[i * 4 + 3] = 1.0f;
      i++;
    }
  }

  if (malformed || i < total_elements) {
    _fill_identity_cube(data, size, dim);
    warn_message("Cube file '%s' is malformed, fallback to identity.", file_name);
  }

  if (dim == 1) {
    texture_fill(texture, size, 1, 1, data, TEXTURE_DATA_TYPE_FP32, 4);
    // Luminary does not have 1D textures right now, map to 2D
    texture->dim = TEXTURE_DIMENSION_TYPE_2D;
  }
  else {
    texture_fill(texture, size, size, size, data, TEXTURE_DATA_TYPE_FP32, 4);
    texture->dim = TEXTURE_DIMENSION_TYPE_3D;
  }

  texture->gamma = 1.0f;

  return LUMINARY_SUCCESS;
}

LuminaryResult image_load(Texture* texture, const char* path) {
  __CHECK_NULL_ARGUMENT(texture);
  __CHECK_NULL_ARGUMENT(path);

  log_message("Loading texture file (%s)", path);

  FILE* file = fopen(path, "rb");

  if (file == (FILE*) 0) {
    __FAILURE_HANDLE(texture_invalidate(texture));
    error_message("File %s could not be opened!", path);
    return LUMINARY_SUCCESS;
  }

  // Block size is very important for performance, it seems that the larger this is the better,
  // however, too large block sizes also means large memory consumption.
  const size_t block_size = 16 * 1024 * 1024;
  size_t file_length      = 0;

  LOCAL uint8_t* file_mem;
  __FAILURE_HANDLE(host_malloc_local(&file_mem, block_size));

  size_t read_size;

  while (read_size = fread(file_mem + file_length, 1, block_size, file), read_size == block_size) {
    file_length += block_size;
    __FAILURE_HANDLE(host_realloc_local(&file_mem, file_length + block_size));
  }

  fclose(file);

  file_length += read_size;

  LuminaryResult result;

  if (_image_is_cube_from_memory(file_mem, file_length)) {
    result = _image_load_cube(texture, file_mem, file_length, path);
  }
  else if (stbi_is_hdr_from_memory(file_mem, (int) file_length)) {
    result = _image_load_hdr(texture, file_mem, file_length, path);
  }
  else if (stbi_is_16_bit_from_memory(file_mem, (int) file_length)) {
    result = _image_load_16(texture, file_mem, file_length, path);
  }
  else {
    result = _image_load_8(texture, file_mem, file_length, path);
  }

  __FAILURE_HANDLE(host_free_local(&file_mem));

  return result;
}

LuminaryResult luminary_image_save(const LuminaryImageSaveArgs* args) {
  __CHECK_NULL_ARGUMENT(args);
  __CHECK_NULL_ARGUMENT(args->file_path);
  __CHECK_NULL_ARGUMENT(args->image.buffer);

  const uint32_t width      = args->image.width;
  const uint32_t height     = args->image.height;
  const uint32_t image_size = width * height * 4;

  uint8_t* buffer;
  __FAILURE_HANDLE(host_malloc((void**) &buffer, image_size));

  uint8_t* buffer_rgb8 = buffer;
  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
      buffer_rgb8[4 * (x + y * width) + 0] = args->image.buffer[4 * (x + y * args->image.ld) + 2];
      buffer_rgb8[4 * (x + y * width) + 1] = args->image.buffer[4 * (x + y * args->image.ld) + 1];
      buffer_rgb8[4 * (x + y * width) + 2] = args->image.buffer[4 * (x + y * args->image.ld) + 0];
      buffer_rgb8[4 * (x + y * width) + 3] = args->image.buffer[4 * (x + y * args->image.ld) + 3];
    }
  }

  const char* file_path_string;
  LuminaryResult apply_result = luminary_path_apply(args->file_path, (const char*) 0, &file_path_string);
  if (apply_result != LUMINARY_SUCCESS) {
    __FAILURE_HANDLE(host_free((void**) &buffer));
    return apply_result;
  }

  int stbi_result = 0;
  switch (args->format) {
    case LUMINARY_IMAGE_SAVE_FORMAT_PNG:
      stbi_result = stbi_write_png(file_path_string, width, height, 4, buffer, width * 4);
      break;
    case LUMINARY_IMAGE_SAVE_FORMAT_JPG:
      stbi_result = stbi_write_jpg(file_path_string, width, height, 4, buffer, args->jpeg_quality);
      break;
    case LUMINARY_IMAGE_SAVE_FORMAT_BMP:
      stbi_result = stbi_write_bmp(file_path_string, width, height, 4, buffer);
      break;
    case LUMINARY_IMAGE_SAVE_FORMAT_TGA:
      stbi_result = stbi_write_tga(file_path_string, width, height, 4, buffer);
      break;
    default:
      break;
  }

  __FAILURE_HANDLE(host_free((void**) &buffer));

  if (!stbi_result) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Failed to save image.");
  }

  return LUMINARY_SUCCESS;
}
