#include "image.h"

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

// Disable warnings about unused static functions in stb_image. We assume that all non-MSVC compilers support GCC style pragmas.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4505)
#else /* _MSC_VER && !__clang__ */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif /* !_MSC_VER || __clang__ */

#include "stb/stb_image.h"

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#else /* _MSC_VER && !__clang__ */
#pragma GCC diagnostic pop
#endif /* !_MSC_VER || __clang__ */

static LuminaryResult _image_load_hdr(Texture* texture, const uint8_t* file_mem, size_t file_length) {
  __CHECK_NULL_ARGUMENT(texture);
  __CHECK_NULL_ARGUMENT(file_mem);

  int width, height, num_components;
  float* data = stbi_loadf_from_memory(file_mem, (int) file_length, &width, &height, &num_components, 4);

  __FAILURE_HANDLE(texture_fill(texture, width, height, 1, data, TEXTURE_DATA_TYPE_FP32, 4));

  // stb_image does not expose this so we have to guess
  texture->gamma = 1.0f;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _image_load_16(Texture* texture, const uint8_t* file_mem, size_t file_length) {
  __CHECK_NULL_ARGUMENT(texture);
  __CHECK_NULL_ARGUMENT(file_mem);

  int width, height, num_components;
  uint16_t* data = stbi_load_16_from_memory(file_mem, (int) file_length, &width, &height, &num_components, 4);

  __FAILURE_HANDLE(texture_fill(texture, width, height, 1, data, TEXTURE_DATA_TYPE_U16, 4));

  // stb_image does not expose this so we have to guess
  texture->gamma = 2.2f;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _image_load_8(Texture* texture, const uint8_t* file_mem, size_t file_length) {
  __CHECK_NULL_ARGUMENT(texture);
  __CHECK_NULL_ARGUMENT(file_mem);

  int width, height, num_components;
  uint8_t* data = stbi_load_from_memory(file_mem, (int) file_length, &width, &height, &num_components, 4);

  __FAILURE_HANDLE(texture_fill(texture, width, height, 1, data, TEXTURE_DATA_TYPE_U8, 4));

  // stb_image does not expose this so we have to guess
  // TODO: Expose this as a per texture setting through the API one day so users can manually fix this.
  texture->gamma = 2.2f;

  return LUMINARY_SUCCESS;
}

LuminaryResult image_load(Texture* texture, const char* path) {
  __CHECK_NULL_ARGUMENT(texture);
  __CHECK_NULL_ARGUMENT(path);

  log_message("Loading texture file (%s)", path);

  FILE* file = fopen(path, "rb");

  if (!file) {
    __FAILURE_HANDLE(texture_invalidate(texture));
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "File %s could not be opened!", path);
  }

  // Block size is very important for performance, it seems that the larger this is the better,
  // however, too large block sizes also means large memory consumption.
  const size_t block_size = 16 * 1024 * 1024;
  size_t file_length      = 0;

  uint8_t* file_mem;
  __FAILURE_HANDLE(host_malloc(&file_mem, block_size));

  size_t read_size;

  while (read_size = fread(file_mem + file_length, 1, block_size, file), read_size == block_size) {
    file_length += block_size;
    __FAILURE_HANDLE(host_realloc(&file_mem, file_length + block_size));
  }

  fclose(file);

  file_length += read_size;

  __FAILURE_HANDLE(host_realloc(&file_mem, file_length));

  if (stbi_is_hdr_from_memory(file_mem, (int) file_length)) {
    __FAILURE_HANDLE(_image_load_hdr(texture, file_mem, file_length));
  }
  else if (stbi_is_16_bit_from_memory(file_mem, (int) file_length)) {
    __FAILURE_HANDLE(_image_load_16(texture, file_mem, file_length));
  }
  else {
    __FAILURE_HANDLE(_image_load_8(texture, file_mem, file_length));
  }

  __FAILURE_HANDLE(host_free(&file_mem));

  return LUMINARY_SUCCESS;
}
