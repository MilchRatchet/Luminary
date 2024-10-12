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

  __FAILURE_HANDLE(sample_count_get_default(&scene->sample_count));
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
      memcpy(&scene->settings, object, sizeof(RendererSettings));
      break;
    case SCENE_ENTITY_CAMERA:
      __FAILURE_HANDLE(camera_check_for_dirty((Camera*) object, &scene->camera, &output_dirty, &integration_dirty));
      scene->flags |= (output_dirty || integration_dirty) ? SCENE_DIRTY_FLAG_CAMERA : 0;
      memcpy(&scene->camera, object, sizeof(Camera));
      break;
    case SCENE_ENTITY_OCEAN:
      __FAILURE_HANDLE(ocean_check_for_dirty((Ocean*) object, &scene->ocean, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_OCEAN : 0;
      memcpy(&scene->ocean, object, sizeof(Ocean));
      break;
    case SCENE_ENTITY_SKY:
      __FAILURE_HANDLE(sky_check_for_dirty((Sky*) object, &scene->sky, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_SKY : 0;
      memcpy(&scene->sky, object, sizeof(Sky));
      break;
    case SCENE_ENTITY_CLOUD:
      __FAILURE_HANDLE(cloud_check_for_dirty((Cloud*) object, &scene->cloud, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_CLOUD : 0;
      memcpy(&scene->cloud, object, sizeof(Cloud));
      break;
    case SCENE_ENTITY_FOG:
      __FAILURE_HANDLE(fog_check_for_dirty((Fog*) object, &scene->fog, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_FOG : 0;
      memcpy(&scene->fog, object, sizeof(Fog));
      break;
    case SCENE_ENTITY_PARTICLES:
      __FAILURE_HANDLE(particles_check_for_dirty((Particles*) object, &scene->particles, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_PARTICLES : 0;
      memcpy(&scene->particles, object, sizeof(Particles));
      break;
    case SCENE_ENTITY_TOY:
      __FAILURE_HANDLE(toy_check_for_dirty((Toy*) object, &scene->toy, &integration_dirty));
      scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_TOY : 0;
      memcpy(&scene->toy, object, sizeof(Toy));
      break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Scene entity does not support scene_update yet.");
  }

  scene->flags |= (output_dirty || integration_dirty) ? SCENE_DIRTY_FLAG_OUTPUT : 0;
  scene->flags |= (integration_dirty) ? SCENE_DIRTY_FLAG_INTEGRATION : 0;
  scene->flags |= (buffers_dirty) ? SCENE_DIRTY_FLAG_BUFFERS : 0;

  __FAILURE_HANDLE(mutex_unlock(scene->mutex));

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
    default:
      __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Scene entity does not support scene_get yet.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_get_locking(Scene* scene, void* object, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(scene);
  __CHECK_NULL_ARGUMENT(object);

  __FAILURE_HANDLE(scene_lock(scene));
  __FAILURE_HANDLE(scene_get(scene, object, entity));
  __FAILURE_HANDLE(scene_unlock(scene));

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_destroy(Scene** scene) {
  __CHECK_NULL_ARGUMENT(scene);

  __FAILURE_HANDLE(mutex_destroy(&(*scene)->mutex));

  __FAILURE_HANDLE(array_destroy(&(*scene)->materials));
  __FAILURE_HANDLE(array_destroy(&(*scene)->instances));

  __FAILURE_HANDLE(host_free(scene));

  return LUMINARY_SUCCESS;
}
