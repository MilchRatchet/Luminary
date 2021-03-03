/*
 * png.c - Store image in png format
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "png.h"
#include "zlib/zlib.h"
#include "texture.h"

static inline void write_int_big_endian(uint8_t* buffer, uint32_t value) {
  buffer[0] = value >> 24;
  buffer[1] = value >> 16;
  buffer[2] = value >> 8;
  buffer[3] = value;
}

static void write_header_to_file(FILE* file) {
  uint8_t* header = (uint8_t*) malloc(8);

  header[0] = 0x89ui8;

  header[1] = 0x50ui8;
  header[2] = 0x4Eui8;
  header[3] = 0x47ui8;

  header[4] = 0x0Dui8;
  header[5] = 0x0Aui8;

  header[6] = 0x1Aui8;

  header[7] = 0x0Aui8;

  fwrite(header, 1, 8, file);

  free(header);
}

static void write_IHDR_chunk_to_file(
  FILE* file, const uint32_t width, const uint32_t height, const uint8_t bit_depth,
  const uint8_t color_type, const uint8_t interlace_method) {
  uint8_t* chunk = (uint8_t*) malloc(25);

  write_int_big_endian(chunk, 13u);

  chunk[4] = 'I';
  chunk[5] = 'H';
  chunk[6] = 'D';
  chunk[7] = 'R';

  write_int_big_endian(chunk + 8, width);
  write_int_big_endian(chunk + 12, height);

  chunk[16] = bit_depth;
  chunk[17] = color_type;
  chunk[18] = 0ui8;
  chunk[19] = 0ui8;
  chunk[20] = interlace_method;

  write_int_big_endian(chunk + 21, (uint32_t) crc32(0, chunk + 4, 17));

  fwrite(chunk, 1, 25, file);

  free(chunk);
}

static void write_IDAT_chunk_to_file(
  FILE* file, const uint8_t* compressed_image, const uint32_t compressed_length) {
  uint8_t* chunk = (uint8_t*) malloc(12 + compressed_length);

  write_int_big_endian(chunk, compressed_length);

  chunk[4] = 'I';
  chunk[5] = 'D';
  chunk[6] = 'A';
  chunk[7] = 'T';

  memcpy(chunk + 8, compressed_image, compressed_length);

  write_int_big_endian(
    chunk + 8 + compressed_length, (uint32_t) crc32(0, chunk + 4, 4 + compressed_length));

  fwrite(chunk, 1, 12 + compressed_length, file);

  free(chunk);
}

static void write_IEND_chunk_to_file(FILE* file) {
  uint8_t* chunk = (uint8_t*) malloc(12);

  write_int_big_endian(chunk, 0u);

  chunk[4] = 'I';
  chunk[5] = 'E';
  chunk[6] = 'N';
  chunk[7] = 'D';

  write_int_big_endian(chunk + 8, (uint32_t) crc32(0, chunk + 4, 4));

  fwrite(chunk, 1ul, 12ul, file);

  free(chunk);
}

/*
 * Does not support filter yet.
 */
int store_as_png(
  const char* filename, const uint8_t* image, const uint32_t image_length, const uint32_t width,
  const uint32_t height, const uint8_t color_type, const uint8_t bit_depth) {
  uint8_t bytes_per_pixel;
  uint8_t bytes_per_channel;

  if (bit_depth == PNG_BITDEPTH_16) {
    bytes_per_channel = 2;
  }
  else {
    bytes_per_channel = 1;
  }

  switch (color_type) {
  case PNG_COLORTYPE_GRAYSCALE:
    bytes_per_pixel = 1 * bytes_per_channel;
    break;

  case PNG_COLORTYPE_TRUECOLOR:
    bytes_per_pixel = 3 * bytes_per_channel;
    break;

  case PNG_COLORTYPE_INDEXED:
    bytes_per_pixel = 1 * bytes_per_channel;
    break;

  case PNG_COLORTYPE_GRAYSCALE_ALPHA:
    bytes_per_pixel = 2 * bytes_per_channel;
    break;

  case PNG_COLORTYPE_TRUECOLOR_ALPHA:
    bytes_per_pixel = 4 * bytes_per_channel;
    break;

  default:
    return 1;
  }

  /* Adding filter byte at the beginning of each scanline */
  uint8_t* filtered_image = (uint8_t*) malloc(image_length + height);
  for (int i = 0; i < height; i++) {
    filtered_image[i * width * bytes_per_pixel + i] = 0;
    memcpy(
      filtered_image + i * width * bytes_per_pixel + i + 1, image + i * width * bytes_per_pixel,
      width * bytes_per_pixel);
  }

  FILE* file = fopen(filename, "wb");

  if (!file) {
    puts("png.c: File could not be opened!");
    return 1;
  }

  write_header_to_file(file);

  write_IHDR_chunk_to_file(file, width, height, bit_depth, color_type, PNG_INTERLACE_OFF);

  uint8_t* compressed_image = (uint8_t*) malloc(image_length + height);

  z_stream defstream;
  defstream.zalloc = Z_NULL;
  defstream.zfree  = Z_NULL;
  defstream.opaque = Z_NULL;

  defstream.avail_in  = (uInt) image_length + height;
  defstream.next_in   = (Bytef*) filtered_image;
  defstream.avail_out = (uInt) image_length + height;
  defstream.next_out  = (Bytef*) compressed_image;

  deflateInit(&defstream, Z_BEST_COMPRESSION);
  deflate(&defstream, Z_FINISH);

  const uLong compressed_length = defstream.total_out;

  deflateEnd(&defstream);

  write_IDAT_chunk_to_file(file, compressed_image, compressed_length);

  write_IEND_chunk_to_file(file);

  fclose(file);

  free(compressed_image);
  free(filtered_image);

  return 0;
}

static inline uint32_t read_int_big_endian(uint8_t* buffer) {
  uint32_t result = 0;

  result += ((uint32_t) buffer[0]) << 24;
  result += ((uint32_t) buffer[1]) << 16;
  result += ((uint32_t) buffer[2]) << 8;
  result += ((uint32_t) buffer[3]);

  return result;
}

static int verify_header(FILE* file) {
  uint8_t* header = (uint8_t*) malloc(8);

  fread(header, 1, 8, file);

  int result = 0;

  result += header[0] ^ 0x89ui8;
  result += header[1] ^ 0x50ui8;
  result += header[2] ^ 0x4Eui8;
  result += header[3] ^ 0x47ui8;
  result += header[4] ^ 0x0Dui8;
  result += header[5] ^ 0x0Aui8;
  result += header[6] ^ 0x1Aui8;
  result += header[7] ^ 0x0Aui8;

  free(header);

  return result;
}

static inline TextureRGBA default_failure() {
  puts("png.c: File content is corrupted!");
  TextureRGBA fallback = {.data = 0, .height = 0, .width = 0};
  return fallback;
}

TextureRGBA load_texture_from_png(const char* filename) {
  FILE* file = fopen(filename, "rb");

  if (!file) {
    puts("png.c: File could not be opened!");
    TextureRGBA fallback = {.data = 0, .height = 0, .width = 0};
    return fallback;
  }

  if (verify_header(file)) {
    puts("png.c: File header does not correspond to png!");
    TextureRGBA fallback = {.data = 0, .height = 0, .width = 0};
    return fallback;
  }

  uint8_t* IHDR = (uint8_t*) malloc(25);

  fread(IHDR, 1, 25, file);

  if (read_int_big_endian(IHDR) != 13u) {
    return default_failure();
  }

  if (read_int_big_endian(IHDR + 4) != 1229472850u) {
    return default_failure();
  }

  const uint32_t width         = read_int_big_endian(IHDR + 8);
  const uint32_t height        = read_int_big_endian(IHDR + 12);
  const uint8_t bit_depth      = IHDR[16];
  const uint8_t color_type     = IHDR[17];
  const uint8_t interlace_type = IHDR[20];

  if (
    color_type != PNG_COLORTYPE_TRUECOLOR_ALPHA || bit_depth != PNG_BITDEPTH_8
    || interlace_type == PNG_INTERLACE_ADAM7) {
    puts("png.c: File properties are not supported!");
    printf("Width: %u, Height: %u\n", width, height);
    printf(
      "bit_depth: %u, color_type: %u, interlace_type: %u\n", bit_depth, color_type, interlace_type);
    TextureRGBA fallback = {.data = 0, .height = 0, .width = 0};
    return fallback;
  }

  if ((uint32_t) crc32(0, IHDR + 4, 17) != read_int_big_endian(IHDR + 21)) {
    return default_failure();
  }

  const uint32_t byte_per_pixel = (bit_depth == PNG_BITDEPTH_16) ? 8u : 4u;

  free(IHDR);

  TextureRGBA result = {.width = width, .height = height};

  uint8_t* chunk = (uint8_t*) malloc(8);

  fread(chunk, 1, 8, file);

  if (read_int_big_endian(chunk + 4) == 1347179589u) {
    puts("png.c: Color Palette found but is ignored!");
    const uint32_t length = read_int_big_endian(chunk);
    fseek(file, length + 4u, SEEK_CUR);

    fread(chunk, 1, 8, file);
  }

  if (read_int_big_endian(chunk + 4) != 1229209940u) {
    puts("png.c: File chunks are unexpected!");
    TextureRGBA fallback = {.data = 0, .height = 0, .width = 0};
    return fallback;
  }

  uint8_t* filtered_data     = (uint8_t*) malloc(width * height * byte_per_pixel + height);
  uint8_t* compressed_buffer = (uint8_t*) malloc(2 * (width * height * byte_per_pixel + height));

  uint32_t combined_compressed_length = 0;

  while (read_int_big_endian(chunk + 4) == 1229209940u) {
    const int compressed_length = read_int_big_endian(chunk);

    uint8_t* compressed_data = (uint8_t*) malloc(compressed_length + 4);

    write_int_big_endian(compressed_data, 1229209940u);

    fread(compressed_data + 4, 1, compressed_length, file);

    fread(chunk, 1, 4, file);

    if ((uint32_t) crc32(0, compressed_data, compressed_length + 4) != read_int_big_endian(chunk)) {
      return default_failure();
    }

    memcpy(compressed_buffer + combined_compressed_length, compressed_data + 4, compressed_length);

    combined_compressed_length += compressed_length;

    free(compressed_data);

    fread(chunk, 1, 8, file);
  }

  free(chunk);

  z_stream defstream;
  defstream.zalloc = Z_NULL;
  defstream.zfree  = Z_NULL;
  defstream.opaque = Z_NULL;

  defstream.avail_in  = (uInt) combined_compressed_length;
  defstream.next_in   = (Bytef*) compressed_buffer;
  defstream.avail_out = (uInt) width * height * byte_per_pixel + height;
  defstream.next_out  = (Bytef*) filtered_data;

  inflateInit(&defstream);
  inflate(&defstream, Z_FINISH);

  inflateEnd(&defstream);

  free(compressed_buffer);

  uint8_t* data = (uint8_t*) malloc(width * height * byte_per_pixel);

  uint8_t filter_error = 0;

  for (uint32_t i = 0; i < height; i++) {
    const uint8_t filter = filtered_data[i * (width * byte_per_pixel + 1)];

    if (filter != 0u && filter != 1u) {
      filter_error++;
      /*printf("Filter Used: %u in Line: %d\n", filter, i);
      for (int k = -10; k < 10; k++) {
        printf("At %d is byte %u\n", k, filtered_data[i * (width * byte_per_pixel + 1) - k]);
      }*/
    }

    memcpy(
      data + i * width * byte_per_pixel, filtered_data + 1 + i * (width * byte_per_pixel + 1),
      width * byte_per_pixel);

    if (filter == 1u) {
      uint8_t predicted_r, predicted_g, predicted_b, predicted_a;

      predicted_r = 0;
      predicted_g = 0;
      predicted_b = 0;
      predicted_a = 0;

      for (uint32_t j = 0; j < width; j++) {
        predicted_r += data[i * width * byte_per_pixel + byte_per_pixel * j];
        predicted_g += data[i * width * byte_per_pixel + byte_per_pixel * j + 1];
        predicted_b += data[i * width * byte_per_pixel + byte_per_pixel * j + 2];
        predicted_a += data[i * width * byte_per_pixel + byte_per_pixel * j + 3];

        data[i * width * byte_per_pixel + byte_per_pixel * j]     = predicted_r;
        data[i * width * byte_per_pixel + byte_per_pixel * j + 1] = predicted_g;
        data[i * width * byte_per_pixel + byte_per_pixel * j + 2] = predicted_b;
        data[i * width * byte_per_pixel + byte_per_pixel * j + 3] = predicted_a;
      }
    }
  }

  if (filter_error) {
    puts("png.c: Unsupported filters were found! Texture may be corrupted!");
  }

  RGBAF* float_data = (RGBAF*) malloc(width * height * sizeof(RGBAF));

  for (uint32_t i = 0; i < width * height; i++) {
    RGBAF pixel = {
      .r = data[i * byte_per_pixel] / 255.0f,
      .g = data[i * byte_per_pixel + 1] / 255.0f,
      .b = data[i * byte_per_pixel + 2] / 255.0f,
      .a = data[i * byte_per_pixel + 3] / 255.0f};

    float_data[i] = pixel;
  }

  free(data);

  result.data = float_data;

  free(filtered_data);

  fclose(file);

  return result;
}
