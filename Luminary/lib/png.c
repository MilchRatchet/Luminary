/*
 * png.c - Store image in png format
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "png.h"
#include "zlib/zlib.h"

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

  if (bit_depth == 16) {
    bytes_per_channel = 2;
  }
  else {
    bytes_per_channel = 1;
  }

  switch (color_type) {
  case 0:
    bytes_per_pixel = 1 * bytes_per_channel;
    break;

  case 2:
    bytes_per_pixel = 3 * bytes_per_channel;
    break;

  case 3:
    bytes_per_pixel = 1 * bytes_per_channel;
    break;

  case 4:
    bytes_per_pixel = 2 * bytes_per_channel;
    break;

  case 6:
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
