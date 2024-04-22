#ifndef LIGHT_H
#define LIGHT_H

#include "utils.h"

void lights_build_set_from_triangles(Scene* scene, TextureRGBA* textures, int dmm_active);
void light_load_ltc_texture(RaytraceInstance* instance);

#endif /* LIGHT_H */
