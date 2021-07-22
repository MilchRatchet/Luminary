#ifndef SCENE_H
#define SCENE_H

#include "primitives.h"
#include "mesh.h"
#include "bvh.h"
#include "texture.h"
#include "utils.h"

Scene load_scene(const char* filename, RaytraceInstance** instance, char** output_name);
void free_scene(Scene scene, RaytraceInstance* instance);

#endif /* SCENE_H */
