#define QOI_IMPLEMENTATION
#include "qoi/qoi.h"

#include "internal_error.h"
#include "qoi.h"
#include "texture.h"
#include "utils.h"

LuminaryResult store_as_qoi(
  const char* filename, const uint8_t* image, const uint32_t width, const uint32_t height, const uint8_t color_type) {
  if (!filename) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Filename is NULL.");
  }

  if (!image) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Image is NULL.");
  }

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
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "QOI image has size 0.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult store_XRGB8_qoi(const char* filename, const XRGB8* image, const int width, const int height) {
  if (!filename) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Filename is NULL.");
  }

  if (!image) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Image is NULL.");
  }

  uint8_t* buffer;
  __FAILURE_HANDLE(host_malloc(&buffer, width * height * sizeof(RGB8)));

  RGB8* buffer_rgb8 = (RGB8*) buffer;
  for (int i = 0; i < height * width; i++) {
    XRGB8 a        = image[i];
    RGB8 result    = {.r = a.r, .g = a.g, .b = a.b};
    buffer_rgb8[i] = result;
  }

  const int ret = store_as_qoi(filename, buffer, width, height, QOI_COLORTYPE_TRUECOLOR);

  __FAILURE_HANDLE(host_free(&buffer));

  if (ret) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "QOI returned error: %d.", ret);
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult qoi_encode_RGBA8(const Texture* tex, int* encoded_size, void** data) {
  __CHECK_NULL_ARGUMENT(tex);
  __CHECK_NULL_ARGUMENT(encoded_size);
  __CHECK_NULL_ARGUMENT(data);

  if (tex->type != TexDataUINT8) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture is not of channel type uint8_t.");
  }

  if (tex->dim != Tex2D) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture is not 2D.");
  }

  const qoi_desc desc = {.width = tex->width, .height = tex->height, .channels = 4, .colorspace = QOI_SRGB};

  *data = qoi_encode(tex->data, &desc, encoded_size);

  return LUMINARY_SUCCESS;
}

LuminaryResult qoi_decode_RGBA8(const void* data, const int size, Texture** texture) {
  __CHECK_NULL_ARGUMENT(data);
  __CHECK_NULL_ARGUMENT(texture);

  qoi_desc desc;
  void* decoded_data = qoi_decode(data, size, &desc, 4);

  __FAILURE_HANDLE(texture_create(texture, desc.width, desc.height, 1, decoded_data, TexDataUINT8, 4));

  return LUMINARY_SUCCESS;
}
