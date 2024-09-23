#ifndef LUMINARY_SCENE_H
#define LUMINARY_SCENE_H

#include "camera.h"
#include "utils.h"

/*
 * The scene struct holds all the data that can be modified at runtime.
 * Scene data is subject to dirty checking.
 * Note that meshes and textures can only be added during runtime but
 * not be modified after the fact and are hence not present here.
 * Instances can be transformed and are hence present here.
 */
struct Scene {
  Camera camera;
} typedef Scene;

LuminaryResult scene_create(Scene** scene);
LuminaryResult scene_destroy(Scene** scene);

#endif /* LUMINARY_SCENE_H */
