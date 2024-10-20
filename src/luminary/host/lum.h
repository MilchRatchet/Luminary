#ifndef LUM_H
#define LUM_H

#include <stdio.h>

#include "utils.h"
#include "wavefront.h"

struct LumFileContent {
  ARRAY char** obj_file_path_strings;
  RendererSettings settings;
  Camera camera;
  Ocean ocean;
  Sky sky;
  Cloud cloud;
  Fog fog;
  Particles particles;
  Toy toy;
  ARRAY MeshInstance* instances;
} typedef LumFileContent;

LuminaryResult lum_content_create(LumFileContent** content);
LuminaryResult lum_read_file(Path* path, LumFileContent* content);
LuminaryResult lum_write_content(Path* path, LumFileContent* content);
LuminaryResult lum_content_destroy(LumFileContent** content);

/*
 * Parses a v5 lum file.
 * @param File File handle to the lum file.
 * @param content LumFileContent instance to which the file's content will be written.
 */
LuminaryResult lum_parse_file_v5(FILE* file, LumFileContent* content);

/*
 * Parses a legacy v4 lum file.
 * @param File File handle to the lum file.
 * @param content LumFileContent instance to which the file's content will be written.
 */
LuminaryResult lum_parse_file_v4(FILE* file, LumFileContent* content);

#endif /* LUM_H */
