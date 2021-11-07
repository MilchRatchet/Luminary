#include "png.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "error.h"
#include "log.h"
#include "texture.h"
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
  FILE* file, const uint32_t width, const uint32_t height, const uint8_t bit_depth, const uint8_t color_type,
  const uint8_t interlace_method) {
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

static void write_IDAT_chunk_to_file(FILE* file, const uint8_t* compressed_image, const uint32_t compressed_length) {
  uint8_t* chunk = (uint8_t*) malloc(12 + compressed_length);

  write_int_big_endian(chunk, compressed_length);

  chunk[4] = 'I';
  chunk[5] = 'D';
  chunk[6] = 'A';
  chunk[7] = 'T';

  memcpy(chunk + 8, compressed_image, compressed_length);

  write_int_big_endian(chunk + 8 + compressed_length, (uint32_t) crc32(0, chunk + 4, 4 + compressed_length));

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
  const char* filename, const uint8_t* image, const uint32_t image_length, const uint32_t width, const uint32_t height,
  const uint8_t color_type, const uint8_t bit_depth) {
  bench_tic();
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
    memcpy(filtered_image + i * width * bytes_per_pixel + i + 1, image + i * width * bytes_per_pixel, width * bytes_per_pixel);
  }

  FILE* file;
  fopen_s(&file, filename, "wb");

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

  bench_toc("Storing PNG");

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

  if (header == (uint8_t*) 0) {
    puts("png.c: Failed to allocate memory!");
    return 1;
  }

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

static inline TextureRGBA default_texture() {
  TextureRGBA fallback = {.data = malloc(sizeof(RGBAF) * 4), .height = 2, .width = 2};
  RGBAF default_pixel  = {.r = 1.0f, .g = 0.0f, .b = 0.0f, .a = 1.0f};
  fallback.data[0]     = default_pixel;
  fallback.data[1]     = default_pixel;
  fallback.data[2]     = default_pixel;
  fallback.data[3]     = default_pixel;
  return fallback;
}

static inline TextureRGBA default_failure() {
  error_message("File content is corrupted!");
  return default_texture();
}

static void reconstruction_1_uint8_RGBA(uint8_t* line, const uint32_t line_length) {
  uint8_t a_r, a_g, a_b, a_a;

  a_r = 0;
  a_g = 0;
  a_b = 0;
  a_a = 0;

  for (uint32_t j = 0; j < line_length; j++) {
    a_r += line[4 * j];
    a_g += line[4 * j + 1];
    a_b += line[4 * j + 2];
    a_a += line[4 * j + 3];

    line[4 * j]     = a_r;
    line[4 * j + 1] = a_g;
    line[4 * j + 2] = a_b;
    line[4 * j + 3] = a_a;
  }
}

static void reconstruction_1_uint8_RGB(uint8_t* line, const uint32_t line_length) {
  uint8_t a_r, a_g, a_b;

  a_r = 0;
  a_g = 0;
  a_b = 0;

  for (uint32_t j = 0; j < line_length; j++) {
    a_r += line[3 * j];
    a_g += line[3 * j + 1];
    a_b += line[3 * j + 2];

    line[3 * j]     = a_r;
    line[3 * j + 1] = a_g;
    line[3 * j + 2] = a_b;
  }
}

static void reconstruction_2_uint8_RGBA(uint8_t* line, const uint32_t line_length, uint8_t* prior_line) {
  uint8_t b_r, b_g, b_b, b_a;

  for (uint32_t j = 0; j < line_length; j++) {
    b_r = prior_line[4 * j];
    b_g = prior_line[4 * j + 1];
    b_b = prior_line[4 * j + 2];
    b_a = prior_line[4 * j + 3];

    line[4 * j] += b_r;
    line[4 * j + 1] += b_g;
    line[4 * j + 2] += b_b;
    line[4 * j + 3] += b_a;
  }
}

static void reconstruction_2_uint8_RGB(uint8_t* line, const uint32_t line_length, uint8_t* prior_line) {
  uint8_t b_r, b_g, b_b;

  for (uint32_t j = 0; j < line_length; j++) {
    b_r = prior_line[3 * j];
    b_g = prior_line[3 * j + 1];
    b_b = prior_line[3 * j + 2];

    line[3 * j] += b_r;
    line[3 * j + 1] += b_g;
    line[3 * j + 2] += b_b;
  }
}

static void reconstruction_3_uint8_RGBA(uint8_t* line, const uint32_t line_length, uint8_t* prior_line) {
  uint8_t a_r, a_g, a_b, a_a;
  uint8_t b_r, b_g, b_b, b_a;

  a_r = 0;
  a_g = 0;
  a_b = 0;
  a_a = 0;

  for (uint32_t j = 0; j < line_length; j++) {
    b_r = prior_line[4 * j];
    b_g = prior_line[4 * j + 1];
    b_b = prior_line[4 * j + 2];
    b_a = prior_line[4 * j + 3];

    line[4 * j] += ((uint16_t) a_r + (uint16_t) b_r) / 2;
    line[4 * j + 1] += ((uint16_t) a_g + (uint16_t) b_g) / 2;
    line[4 * j + 2] += ((uint16_t) a_b + (uint16_t) b_b) / 2;
    line[4 * j + 3] += ((uint16_t) a_a + (uint16_t) b_a) / 2;

    a_r = line[4 * j];
    a_g = line[4 * j + 1];
    a_b = line[4 * j + 2];
    a_a = line[4 * j + 3];
  }
}

static void reconstruction_3_uint8_RGB(uint8_t* line, const uint32_t line_length, uint8_t* prior_line) {
  uint8_t a_r, a_g, a_b;
  uint8_t b_r, b_g, b_b;

  a_r = 0;
  a_g = 0;
  a_b = 0;

  for (uint32_t j = 0; j < line_length; j++) {
    b_r = prior_line[3 * j];
    b_g = prior_line[3 * j + 1];
    b_b = prior_line[3 * j + 2];

    line[3 * j] += ((uint16_t) a_r + (uint16_t) b_r) / 2;
    line[3 * j + 1] += ((uint16_t) a_g + (uint16_t) b_g) / 2;
    line[3 * j + 2] += ((uint16_t) a_b + (uint16_t) b_b) / 2;

    a_r = line[3 * j];
    a_g = line[3 * j + 1];
    a_b = line[3 * j + 2];
  }
}

static uint8_t paeth_uint8(uint8_t a, uint8_t b, uint8_t c) {
  uint8_t pr;
  const int16_t p  = (int16_t) a + (int16_t) b - (int16_t) c;
  const int16_t pa = abs(p - (int16_t) a);
  const int16_t pb = abs(p - (int16_t) b);
  const int16_t pc = abs(p - (int16_t) c);
  if (pa <= pb && pa <= pc) {
    pr = a;
  }
  else if (pb <= pc) {
    pr = b;
  }
  else {
    pr = c;
  }
  return pr;
}

static void reconstruction_4_uint8_RGBA(uint8_t* line, const uint32_t line_length, uint8_t* prior_line) {
  uint8_t a_r, a_g, a_b, a_a;
  uint8_t b_r, b_g, b_b, b_a;
  uint8_t c_r, c_g, c_b, c_a;

  a_r = 0;
  a_g = 0;
  a_b = 0;
  a_a = 0;

  c_r = 0;
  c_g = 0;
  c_b = 0;
  c_a = 0;

  for (uint32_t j = 0; j < line_length; j++) {
    b_r = prior_line[4 * j];
    b_g = prior_line[4 * j + 1];
    b_b = prior_line[4 * j + 2];
    b_a = prior_line[4 * j + 3];

    line[4 * j] += paeth_uint8(a_r, b_r, c_r);
    line[4 * j + 1] += paeth_uint8(a_g, b_g, c_g);
    line[4 * j + 2] += paeth_uint8(a_b, b_b, c_b);
    line[4 * j + 3] += paeth_uint8(a_a, b_a, c_a);

    a_r = line[4 * j];
    a_g = line[4 * j + 1];
    a_b = line[4 * j + 2];
    a_a = line[4 * j + 3];

    c_r = prior_line[4 * j];
    c_g = prior_line[4 * j + 1];
    c_b = prior_line[4 * j + 2];
    c_a = prior_line[4 * j + 3];
  }
}

static void reconstruction_4_uint8_RGB(uint8_t* line, const uint32_t line_length, uint8_t* prior_line) {
  uint8_t a_r, a_g, a_b;
  uint8_t b_r, b_g, b_b;
  uint8_t c_r, c_g, c_b;

  a_r = 0;
  a_g = 0;
  a_b = 0;

  c_r = 0;
  c_g = 0;
  c_b = 0;

  for (uint32_t j = 0; j < line_length; j++) {
    b_r = prior_line[3 * j];
    b_g = prior_line[3 * j + 1];
    b_b = prior_line[3 * j + 2];

    line[3 * j] += paeth_uint8(a_r, b_r, c_r);
    line[3 * j + 1] += paeth_uint8(a_g, b_g, c_g);
    line[3 * j + 2] += paeth_uint8(a_b, b_b, c_b);

    a_r = line[3 * j];
    a_g = line[3 * j + 1];
    a_b = line[3 * j + 2];

    c_r = prior_line[3 * j];
    c_g = prior_line[3 * j + 1];
    c_b = prior_line[3 * j + 2];
  }
}

TextureRGBA load_texture_from_png(const char* filename) {
  FILE* file;
  fopen_s(&file, filename, "rb");

  if (!file) {
    error_message("File could not be opened!");
    return default_texture();
  }

  if (verify_header(file)) {
    error_message("File header does not correspond to png!");
    fclose(file);
    return default_texture();
  }

  uint8_t* IHDR = (uint8_t*) malloc(25);

  if (IHDR == (uint8_t*) 0) {
    error_message("Failed to allocate memory!");
    fclose(file);
    return default_texture();
  }

  fread(IHDR, 1, 25, file);

  if (read_int_big_endian(IHDR) != 13u) {
    free(IHDR);
    fclose(file);
    return default_failure();
  }

  if (read_int_big_endian(IHDR + 4) != 1229472850u) {
    free(IHDR);
    fclose(file);
    return default_failure();
  }

  const uint32_t width         = read_int_big_endian(IHDR + 8);
  const uint32_t height        = read_int_big_endian(IHDR + 12);
  const uint8_t bit_depth      = IHDR[16];
  const uint8_t color_type     = IHDR[17];
  const uint8_t interlace_type = IHDR[20];

  if (color_type != PNG_COLORTYPE_TRUECOLOR_ALPHA && color_type != PNG_COLORTYPE_TRUECOLOR) {
    free(IHDR);
    error_message("Texture is not in RGB/RGBA format!");
    fclose(file);
    return default_texture();
  }

  if (bit_depth != PNG_BITDEPTH_8) {
    free(IHDR);
    error_message("Texture does not have 8 bit depth!");
    fclose(file);
    return default_texture();
  }

  if (interlace_type == PNG_INTERLACE_ADAM7) {
    free(IHDR);
    error_message("Interlaced textures are not supported!");
    fclose(file);
    return default_texture();
  }

  if ((uint32_t) crc32(0, IHDR + 4, 17) != read_int_big_endian(IHDR + 21)) {
    free(IHDR);
    error_message("Texture is corrupted!");
    fclose(file);
    return default_failure();
  }

  const uint32_t byte_per_pixel = (color_type == PNG_COLORTYPE_TRUECOLOR) ? 3u : 4u;

  free(IHDR);

  TextureRGBA result = {.width = width, .height = height};

  uint8_t* chunk = (uint8_t*) malloc(8);

  if (chunk == (uint8_t*) 0) {
    error_message("Failed to allocate memory!");
    return default_texture();
  }

  uint8_t* filtered_data     = (uint8_t*) malloc(width * height * byte_per_pixel + height);
  uint8_t* compressed_buffer = (uint8_t*) malloc(2 * (width * height * byte_per_pixel + height));

  uint32_t combined_compressed_length = 0;

  int data_left = (fread(chunk, 1, 8, file) == 8);

  while (data_left) {
    const int length = read_int_big_endian(chunk);

    if (read_int_big_endian(chunk + 4) == 1229209940u) {
      uint8_t* compressed_data = (uint8_t*) malloc(length + 4);

      write_int_big_endian(compressed_data, 1229209940u);

      fread(compressed_data + 4, 1, length, file);

      fread(chunk, 1, 4, file);

      if ((uint32_t) crc32(0, compressed_data, length + 4) != read_int_big_endian(chunk)) {
        return default_failure();
      }

      memcpy(compressed_buffer + combined_compressed_length, compressed_data + 4, length);

      combined_compressed_length += length;

      free(compressed_data);
    }
    else {
      fseek(file, length + 4u, SEEK_CUR);
    }

    data_left = (fread(chunk, 1, 8, file) == 8);
  }

  fclose(file);
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

  uint8_t* line_buffer    = (uint8_t*) malloc(width * byte_per_pixel);
  uint8_t* buffer_address = line_buffer;

  if (line_buffer == (uint8_t*) 0) {
    error_message("Failed to allocate memory!");
    free(data);
    free(filtered_data);
    return default_texture();
  }

  memset(line_buffer, 0, width * byte_per_pixel);

  for (uint32_t i = 0; i < height; i++) {
    const uint8_t filter = filtered_data[i * (width * byte_per_pixel + 1)];

    memcpy(data + i * width * byte_per_pixel, filtered_data + 1 + i * (width * byte_per_pixel + 1), width * byte_per_pixel);

    if (color_type == PNG_COLORTYPE_TRUECOLOR) {
      if (filter == 1u) {
        reconstruction_1_uint8_RGB(data + i * width * byte_per_pixel, width);
      }
      else if (filter == 2u) {
        reconstruction_2_uint8_RGB(data + i * width * byte_per_pixel, width, line_buffer);
      }
      else if (filter == 3u) {
        reconstruction_3_uint8_RGB(data + i * width * byte_per_pixel, width, line_buffer);
      }
      else if (filter == 4u) {
        reconstruction_4_uint8_RGB(data + i * width * byte_per_pixel, width, line_buffer);
      }
    }
    else {
      if (filter == 1u) {
        reconstruction_1_uint8_RGBA(data + i * width * byte_per_pixel, width);
      }
      else if (filter == 2u) {
        reconstruction_2_uint8_RGBA(data + i * width * byte_per_pixel, width, line_buffer);
      }
      else if (filter == 3u) {
        reconstruction_3_uint8_RGBA(data + i * width * byte_per_pixel, width, line_buffer);
      }
      else if (filter == 4u) {
        reconstruction_4_uint8_RGBA(data + i * width * byte_per_pixel, width, line_buffer);
      }
    }

    line_buffer = data + i * width * byte_per_pixel;
  }

  free(buffer_address);

  RGBAF* float_data = (RGBAF*) malloc(width * height * sizeof(RGBAF));

  if (color_type == PNG_COLORTYPE_TRUECOLOR) {
    for (uint32_t i = 0; i < width * height; i++) {
      RGBAF pixel = {
        .r = data[i * byte_per_pixel] / 255.0f,
        .g = data[i * byte_per_pixel + 1] / 255.0f,
        .b = data[i * byte_per_pixel + 2] / 255.0f,
        .a = 1.0f};

      float_data[i] = pixel;
    }
  }
  else {
    for (uint32_t i = 0; i < width * height; i++) {
      RGBAF pixel = {
        .r = data[i * byte_per_pixel] / 255.0f,
        .g = data[i * byte_per_pixel + 1] / 255.0f,
        .b = data[i * byte_per_pixel + 2] / 255.0f,
        .a = data[i * byte_per_pixel + 3] / 255.0f};

      float_data[i] = pixel;
    }
  }

  free(data);

  result.data = float_data;

  free(filtered_data);

  return result;
}

int store_XRGB8_png(const char* filename, const XRGB8* image, const int width, const int height) {
  uint8_t* buffer = (uint8_t*) malloc(width * height * 3);

  RGB8* buffer_rgb8 = (RGB8*) buffer;
  for (int i = 0; i < height * width; i++) {
    XRGB8 a        = image[i];
    RGB8 result    = {.r = a.r, .g = a.g, .b = a.b};
    buffer_rgb8[i] = result;
  }

  int ret = store_as_png(filename, buffer, width * height * 3, width, height, PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);

  free(buffer);

  return ret;
}
