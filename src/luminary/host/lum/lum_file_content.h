#ifndef LUMINARY_LUM_FILE_CONTENT_H
#define LUMINARY_LUM_FILE_CONTENT_H

#include "host/wavefront.h"
#include "utils.h"

struct LumFileContent {
  ARRAY char** obj_file_path_strings;
  WavefrontArguments wavefront_args;
  RendererSettings settings;
  Camera camera;
  Ocean ocean;
  Sky sky;
  Cloud cloud;
  Fog fog;
  Particles particles;
  ARRAY MeshInstance* instances;
} typedef LumFileContent;

LuminaryResult lum_file_content_create(LumFileContent** content);
LuminaryResult lum_file_content_apply(LumFileContent* content, LuminaryHost* host, const Path* base_path);
LuminaryResult lum_file_content_destroy(LumFileContent** content);

#endif /* LUMINARY_LUM_FILE_CONTENT_H */
