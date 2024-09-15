#define QOI_IMPLEMENTATION
#include "qoi/qoi.h"

#include "log.h"
#include "qoi.h"
#include "texture.h"

int store_as_qoi(const char* filename, const uint8_t* image, const uint32_t width, const uint32_t height, const uint8_t color_type) {
  char channels;

  switch (color_type) {
    case QOI_COLORTYPE_TRUECOLOR:
      channels = 3;
      break;
    case QOI_COLORTYPE_TRUECOLOR_ALPHA:
      channels = 4;
      break;
    default:
      error_message("Color type %u is not a valid QOI color type.", color_type);
      return 1;
  }

  log_message("Storing qoi file (%s) Size: %dx%d Channels: %d", filename, width, height, channels);

  qoi_desc desc = {.width = width, .height = height, .channels = channels, .colorspace = QOI_SRGB};

  const int size = qoi_write(filename, image, &desc);

  if (!size) {
    error_message("QOI image has size 0.");
    return 1;
  }

  return 0;
}

int store_XRGB8_qoi(const char* filename, const XRGB8* image, const int width, const int height) {
  uint8_t* buffer = (uint8_t*) malloc(width * height * 3);

  RGB8* buffer_rgb8 = (RGB8*) buffer;
  for (int i = 0; i < height * width; i++) {
    XRGB8 a        = image[i];
    RGB8 result    = {.r = a.r, .g = a.g, .b = a.b};
    buffer_rgb8[i] = result;
  }

  int ret = store_as_qoi(filename, buffer, width, height, QOI_COLORTYPE_TRUECOLOR);

  free(buffer);

  return ret;
}

void* qoi_encode_RGBA8(const Texture* tex, int* encoded_size) {
  if (!encoded_size || !tex) {
    return (void*) 0;
  }

  if (tex->type != TexDataUINT8) {
    return (void*) 0;
  }

  if (tex->storage != TexStorageCPU) {
    return (void*) 0;
  }

  if (tex->dim != Tex2D) {
    return (void*) 0;
  }

  const qoi_desc desc = {.width = tex->pitch, .height = tex->height, .channels = 4, .colorspace = QOI_SRGB};

  return qoi_encode(tex->data, &desc, encoded_size);
}

Texture* qoi_decode_RGBA8(const void* data, const int size) {
  if (!data) {
    return (Texture*) 0;
  }

  qoi_desc desc;
  void* decoded_data = qoi_decode(data, size, &desc, 4);

  Texture* tex = malloc(sizeof(Texture));
  texture_create(tex, desc.width, desc.height, 1, desc.width, decoded_data, TexDataUINT8, 4, TexStorageCPU);

  return tex;
}
