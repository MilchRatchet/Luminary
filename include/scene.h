#ifndef SCENE_H
#define SCENE_H

#include "bvh.h"
#include "structs.h"
#include "utils.h"

RaytraceInstance* load_scene(const char* filename);
RaytraceInstance* load_obj_as_scene(char* filename);
void serialize_scene(RaytraceInstance* instance);
void free_atlases(RaytraceInstance* instance);
void free_strings(RaytraceInstance* instance);
void free_scene(Scene scene);

#endif /* SCENE_H */
