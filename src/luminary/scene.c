#include "scene.h"

#include <string.h>

#include "camera.h"
#include "cloud.h"
#include "fog.h"
#include "internal_error.h"
#include "ocean.h"
#include "particles.h"
#include "settings.h"
#include "sky.h"
#include "toy.h"

LuminaryResult scene_create(Scene** _scene) {
  __CHECK_NULL_ARGUMENT(_scene);

  Scene* scene;
  __FAILURE_HANDLE(host_malloc(&scene, sizeof(Scene)));

  __FAILURE_HANDLE(mutex_create(&scene->mutex));

  __FAILURE_HANDLE(settings_get_default(&scene->settings));
  __FAILURE_HANDLE(camera_get_default(&scene->camera));
  __FAILURE_HANDLE(ocean_get_default(&scene->ocean));
  __FAILURE_HANDLE(sky_get_default(&scene->sky));
  __FAILURE_HANDLE(cloud_get_default(&scene->cloud));
  __FAILURE_HANDLE(fog_get_default(&scene->fog));
  __FAILURE_HANDLE(particles_get_default(&scene->particles));
  __FAILURE_HANDLE(toy_get_default(&scene->toy));

  __FAILURE_HANDLE(array_create(&scene->materials, sizeof(Material), 16));
  __FAILURE_HANDLE(array_create(&scene->instances, sizeof(MeshInstance), 16));

  __FAILURE_HANDLE(host_malloc(&scene->scratch_buffer, 4096));

  scene->flags = 0xFFFFFFFFu;

  *_scene = scene;

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_lock(Scene* scene) {
  __CHECK_NULL_ARGUMENT(scene);

  __FAILURE_HANDLE(mutex_lock(scene->mutex));

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_get_dirty_flags(const Scene* scene, SceneDirtyFlags* flags) {
  __CHECK_NULL_ARGUMENT(scene);
  __CHECK_NULL_ARGUMENT(flags);

  *flags = scene->flags;

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_unlock(Scene* scene) {
  __CHECK_NULL_ARGUMENT(scene);

  __FAILURE_HANDLE(mutex_unlock(scene->mutex));

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_update(Scene* scene, const void* object, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(scene);
  __CHECK_NULL_ARGUMENT(object);

  __FAILURE_HANDLE(mutex_lock(scene->mutex));

  bool output_dirty      = false;
  bool integration_dirty = false;
  bool buffers_dirty     = false;

  switch (entity) {
    case SCENE_ENTITY_SETTINGS:
      __FAILURE_HANDLE(settings_check_for_dirty((RendererSettings*) object, &scene->settings, &integration_dirty, &buffers_dirty));
      scene->flags |= (integration_dirty || buffers_dirty) ? SCENE_DIRTY_FLAG_SETTINGS : 0;
      break;
    case SCENE_ENTITY_CAMERA:
      __FAILURE_HANDLE(camera_check_for_dirty((Camera*) object, &scene->camera, &output_dirty, &integration_dirty));
      scene->flags |= (output_dirty || integration_dirty) ? SCENE_DIRTY_FLAG_CAMERA : 0;
      break;
    case SCENE_ENTITY_OCEAN:
      __FAILURE_HANDLE(ocean_check_for_dirty((Ocean*) object, &scene->ocean, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_OCEAN : 0;
      break;
    case SCENE_ENTITY_SKY:
      __FAILURE_HANDLE(sky_check_for_dirty((Sky*) object, &scene->sky, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_SKY : 0;
      break;
    case SCENE_ENTITY_CLOUD:
      __FAILURE_HANDLE(cloud_check_for_dirty((Cloud*) object, &scene->cloud, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_CLOUD : 0;
      break;
    case SCENE_ENTITY_FOG:
      __FAILURE_HANDLE(fog_check_for_dirty((Fog*) object, &scene->fog, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_FOG : 0;
      break;
    case SCENE_ENTITY_PARTICLES:
      __FAILURE_HANDLE(particles_check_for_dirty((Particles*) object, &scene->particles, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_PARTICLES : 0;
      break;
    case SCENE_ENTITY_TOY:
      __FAILURE_HANDLE(toy_check_for_dirty((Toy*) object, &scene->toy, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_TOY : 0;
      break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Scene entity does not support scene_update yet.");
  }

  scene->flags |= (output_dirty || integration_dirty) ? SCENE_DIRTY_FLAG_OUTPUT : 0;
  scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_INTEGRATION : 0;
  scene->flags |= (buffers_dirty) ? SCENE_DIRTY_FLAG_BUFFERS : 0;

  if (output_dirty || integration_dirty || buffers_dirty) {
    __FAILURE_HANDLE(scene_update_force(scene, object, entity));
  }

  __FAILURE_HANDLE(mutex_unlock(scene->mutex));

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_update_force(Scene* scene, const void* object, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(scene);
  __CHECK_NULL_ARGUMENT(object);

  switch (entity) {
    case SCENE_ENTITY_SETTINGS:
      memcpy(&scene->settings, object, sizeof(RendererSettings));
      break;
    case SCENE_ENTITY_CAMERA:
      memcpy(&scene->camera, object, sizeof(Camera));
      break;
    case SCENE_ENTITY_OCEAN:
      memcpy(&scene->ocean, object, sizeof(Ocean));
      break;
    case SCENE_ENTITY_SKY:
      memcpy(&scene->sky, object, sizeof(Sky));
      break;
    case SCENE_ENTITY_CLOUD:
      memcpy(&scene->cloud, object, sizeof(Cloud));
      break;
    case SCENE_ENTITY_FOG:
      memcpy(&scene->fog, object, sizeof(Fog));
      break;
    case SCENE_ENTITY_PARTICLES:
      memcpy(&scene->particles, object, sizeof(Particles));
      break;
    case SCENE_ENTITY_TOY:
      memcpy(&scene->toy, object, sizeof(Toy));
      break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Scene entity does not support scene_update yet.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_get(Scene* scene, void* object, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(scene);
  __CHECK_NULL_ARGUMENT(object);

  switch (entity) {
    case SCENE_ENTITY_SETTINGS:
      memcpy(object, &scene->settings, sizeof(RendererSettings));
      break;
    case SCENE_ENTITY_CAMERA:
      memcpy(object, &scene->camera, sizeof(Camera));
      break;
    case SCENE_ENTITY_OCEAN:
      memcpy(object, &scene->ocean, sizeof(Ocean));
      break;
    case SCENE_ENTITY_SKY:
      memcpy(object, &scene->sky, sizeof(Sky));
      break;
    case SCENE_ENTITY_CLOUD:
      memcpy(object, &scene->cloud, sizeof(Cloud));
      break;
    case SCENE_ENTITY_FOG:
      memcpy(object, &scene->fog, sizeof(Fog));
      break;
    case SCENE_ENTITY_PARTICLES:
      memcpy(object, &scene->particles, sizeof(Particles));
      break;
    case SCENE_ENTITY_TOY:
      memcpy(object, &scene->toy, sizeof(Toy));
      break;
    case SCENE_ENTITY_MATERIALS:
      memcpy(object, &scene->materials, sizeof(ARRAY Material*));
      break;
    case SCENE_ENTITY_INSTANCES:
      memcpy(object, &scene->instances, sizeof(ARRAY MeshInstance*));
      break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Scene entity does not support scene_get yet.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_get_locking(Scene* scene, void* object, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(scene);
  __CHECK_NULL_ARGUMENT(object);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(scene_lock(scene));
  __FAILURE_HANDLE_CRITICAL(scene_get(scene, object, entity));
  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(scene_unlock(scene));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_propagate_changes(Scene* scene, Scene* src) {
  __CHECK_NULL_ARGUMENT(scene);
  __CHECK_NULL_ARGUMENT(src);

  // This cannot deadlock because propagation only happens caller->host->device.
  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(scene_lock(scene));
  __FAILURE_HANDLE_CRITICAL(scene_lock(src));

  uint64_t current_entity = SCENE_ENTITY_SETTINGS;
  while (src->flags && current_entity < SCENE_ENTITY_COUNT) {
    if (src->flags & SCENE_ENTITY_TO_DIRTY(current_entity)) {
      __FAILURE_HANDLE_CRITICAL(scene_get(src, scene->scratch_buffer, current_entity));
      __FAILURE_HANDLE_CRITICAL(scene_update_force(scene, scene->scratch_buffer, current_entity));
    }

    current_entity++;
  }

  scene->flags |= src->flags;
  src->flags = 0;

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(scene_unlock(src));
  __FAILURE_HANDLE(scene_unlock(scene));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_destroy(Scene** scene) {
  __CHECK_NULL_ARGUMENT(scene);

  __FAILURE_HANDLE(mutex_destroy(&(*scene)->mutex));

  __FAILURE_HANDLE(array_destroy(&(*scene)->materials));
  __FAILURE_HANDLE(array_destroy(&(*scene)->instances));

  __FAILURE_HANDLE(host_free(&(*scene)->scratch_buffer));

  __FAILURE_HANDLE(host_free(scene));

  return LUMINARY_SUCCESS;
}
