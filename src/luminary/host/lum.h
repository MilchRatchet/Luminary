#ifndef LUM_H
#define LUM_H

#include <stdio.h>

#include "utils.h"

#define LUM_FILE_MAJOR_VERSION_INVALID (0xFFFFFFFF)

struct LumFile {
  Path* path;
  uint32_t parsed_major_version;
  void* parsed_data;
} typedef LumFile;

LuminaryResult lum_file_create(LumFile** file);
LuminaryResult lum_file_parse(LumFile* file, Path* path);
LuminaryResult lum_file_apply(LumFile* file, LuminaryHost* host);
LuminaryResult lum_file_destroy(LumFile** file);

#endif /* LUM_H */
