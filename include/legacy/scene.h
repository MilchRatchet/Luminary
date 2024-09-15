#ifndef SCENE_H
#define SCENE_H

#include "bvh.h"
#include "structs.h"
#include "utils.h"
#include "wavefront.h"

void scene_init(Scene** _scene);
void scene_create_from_wavefront(Scene* scene, WavefrontContent* content);
RaytraceInstance* scene_load_lum(const char* filename, CommandlineOptions options);
RaytraceInstance* scene_load_obj(char* filename, CommandlineOptions options);
void scene_serialize(RaytraceInstance* instance);
void free_atlases(RaytraceInstance* instance);
void free_strings(RaytraceInstance* instance);
void scene_clear(Scene** scene);

#endif /* SCENE_H */
