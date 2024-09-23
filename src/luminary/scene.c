#include "scene.h"

#include "internal_error.h"

LuminaryResult scene_create(Scene** _scene) {
  __CHECK_NULL_ARGUMENT(_scene);

  Scene* scene;
  __FAILURE_HANDLE(host_malloc(&scene, sizeof(Scene)));

  __FAILURE_HANDLE(camera_get_default(&scene->camera));

  *_scene = scene;

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_destroy(Scene** scene) {
  __CHECK_NULL_ARGUMENT(scene);

  __FAILURE_HANDLE(host_free(scene));

  return LUMINARY_SUCCESS;
}
