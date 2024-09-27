#include "scene.h"

#include "internal_error.h"

LuminaryResult scene_create(Scene** _scene) {
  __CHECK_NULL_ARGUMENT(_scene);

  Scene* scene;
  __FAILURE_HANDLE(host_malloc(&scene, sizeof(Scene)));

  __FAILURE_HANDLE(mutex_create(&scene->mutex));

  __FAILURE_HANDLE(sample_count_get_default(&scene->sample_count));
  __FAILURE_HANDLE(camera_get_default(&scene->camera));

  __FAILURE_HANDLE(array_create(&scene->instances, sizeof(Instance), 16));
  __FAILURE_HANDLE(array_create(&scene->instance_updates, sizeof(InstanceUpdate), 16));

  *_scene = scene;

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_destroy(Scene** scene) {
  __CHECK_NULL_ARGUMENT(scene);

  __FAILURE_HANDLE(mutex_destroy(&(*scene)->mutex));

  __FAILURE_HANDLE(array_destroy(&(*scene)->instances));
  __FAILURE_HANDLE(array_destroy(&(*scene)->instance_updates));

  __FAILURE_HANDLE(host_free(scene));

  return LUMINARY_SUCCESS;
}
