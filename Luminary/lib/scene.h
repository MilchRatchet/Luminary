#ifndef SCENE_H
#define SCENE_H

#include "bvh.h"
#include "mesh.h"
#include "primitives.h"
#include "texture.h"
#include "utils.h"

Scene load_scene(const char* filename, RaytraceInstance** instance);
void serialize_scene(RaytraceInstance* instance);
void free_scene(Scene scene, RaytraceInstance* instance);

#endif /* SCENE_H */
