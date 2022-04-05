#define QOI_IMPLEMENTATION
#include "qoi/qoi.h"

#include "bench.h"
#include "log.h"
#include "qoi.h"

int store_as_qoi(const char* filename, const uint8_t* image, const uint32_t width, const uint32_t height, const uint8_t color_type) {
  bench_tic();
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

  bench_toc("Storing QOI");

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
