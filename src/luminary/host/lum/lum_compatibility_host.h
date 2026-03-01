#ifndef LUMINARY_LUM_COMPATIBILITY_HOST_H
#define LUMINARY_LUM_COMPATIBILITY_HOST_H

#include "dictionary.h"
#include "lum_builtins.h"
#include "mesh.h"
#include "texture.h"
#include "utils.h"

struct LumCompatibilityHost {
  uint32_t version;
  LumBuiltinSettings settings;
  LumBuiltinCamera camera;
  LumBuiltinOcean ocean;
  LumBuiltinSky sky;
  LumBuiltinCloud cloud;
  LumBuiltinFog fog;
  LumBuiltinParticles particles;
  ARRAY Mesh** meshes;
  ARRAY LumBuiltinMaterial* materials;
  ARRAY LumBuiltinInstance* mesh_instances;
  ARRAY Texture** textures;
  Dictionary* mesh_instance_name_dict;
  Dictionary* material_name_dict;
  Dictionary* mesh_name_dict;
  Dictionary* texture_name_dict;
} typedef LumCompatibilityHost;

LuminaryResult lum_compatibility_host_create(LumCompatibilityHost** host);
LuminaryResult lum_compatibility_host_init(LumCompatibilityHost* host, uint32_t version);
LuminaryResult lum_compatibility_host_apply(LumCompatibilityHost* host, LuminaryHost* dst_host);
LuminaryResult lum_compatibility_host_destroy(LumCompatibilityHost** host);

#endif /* LUMINARY_LUM_COMPATIBILITY_HOST_H */
