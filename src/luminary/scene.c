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

  __FAILURE_HANDLE(array_create(&scene->material_updates, sizeof(MaterialUpdate), 16));
  __FAILURE_HANDLE(array_create(&scene->instance_updates, sizeof(MeshInstanceUpdate), 16));

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

  scene->flags |= src->flags;

  uint64_t current_entity = SCENE_ENTITY_GLOBAL_START;
  while (src->flags && current_entity <= SCENE_ENTITY_GLOBAL_END) {
    const uint32_t entity_dirty_flag = SCENE_ENTITY_TO_DIRTY(current_entity);
    if (src->flags & entity_dirty_flag) {
      __FAILURE_HANDLE_CRITICAL(scene_get(src, scene->scratch_buffer, current_entity));
      __FAILURE_HANDLE_CRITICAL(scene_update_force(scene, scene->scratch_buffer, current_entity));
      src->flags &= ~entity_dirty_flag;
    }

    current_entity++;
  }

  current_entity = SCENE_ENTITY_LIST_START;
  while (src->flags && current_entity <= SCENE_ENTITY_LIST_END) {
    const uint32_t entity_dirty_flag = SCENE_ENTITY_TO_DIRTY(current_entity);
    if (src->flags & entity_dirty_flag) {
      switch (current_entity) {
        case SCENE_ENTITY_MATERIALS:
          __FAILURE_HANDLE_CRITICAL(array_append(&scene->material_updates, src->material_updates));
          break;
        case SCENE_ENTITY_INSTANCES:
          __FAILURE_HANDLE_CRITICAL(array_append(&scene->instance_updates, src->instance_updates));
          break;
        default:
          __RETURN_ERROR_CRITICAL(LUMINARY_ERROR_API_EXCEPTION, "Entity is not a list entity.");
      }

      __FAILURE_HANDLE_CRITICAL(scene_apply_list_changes(src, current_entity));

      src->flags &= ~entity_dirty_flag;
    }

    current_entity++;
  }

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(scene_unlock(src));
  __FAILURE_HANDLE(scene_unlock(scene));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_apply_list_changes(Scene* scene, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(scene);

  switch (entity) {
    case SCENE_ENTITY_MATERIALS: {
      uint32_t material_updates_count;
      __FAILURE_HANDLE(array_get_num_elements(scene->material_updates, &material_updates_count));

      uint32_t material_count;
      __FAILURE_HANDLE(array_get_num_elements(scene->materials, &material_count));

      for (uint32_t material_update_id = 0; material_update_id < material_updates_count; material_update_id++) {
        const MaterialUpdate update = scene->material_updates[material_update_id];

        // New element
        if (update.material_id >= material_count) {
          __FAILURE_HANDLE(array_push(&scene->materials, &update.material));
          continue;
        }

        scene->materials[update.material_id] = update.material;
      }

      __FAILURE_HANDLE(array_clear(scene->material_updates));
    } break;
    case SCENE_ENTITY_INSTANCES: {
      uint32_t instance_updates_count;
      __FAILURE_HANDLE(array_get_num_elements(scene->instance_updates, &instance_updates_count));

      uint32_t instance_count;
      __FAILURE_HANDLE(array_get_num_elements(scene->instances, &instance_count));

      for (uint32_t instance_update_id = 0; instance_update_id < instance_updates_count; instance_update_id++) {
        const MeshInstanceUpdate update = scene->instance_updates[instance_update_id];

        // New element
        if (update.instance_id >= instance_count) {
          __FAILURE_HANDLE(array_push(&scene->instances, &update.instance));
          continue;
        }

        scene->instances[update.instance_id] = update.instance;
      }

      __FAILURE_HANDLE(array_clear(scene->instance_updates));
    } break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Entity is not a list entity.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_add_entry(Scene* scene, const void* object, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(scene);
  __CHECK_NULL_ARGUMENT(object);

  switch (entity) {
    case SCENE_ENTITY_MATERIALS: {
      // Get canonical new material ID.
      uint32_t material_id;
      __FAILURE_HANDLE(array_get_num_elements(scene->materials, &material_id));

      // There could be new materials queued in the material_updates, we need to check for that.
      uint32_t material_updates_count;
      __FAILURE_HANDLE(array_get_num_elements(scene->material_updates, &material_updates_count));

      for (uint32_t material_update_id = 0; material_update_id < material_updates_count; material_update_id++) {
        const uint32_t material_id_of_update = scene->material_updates[material_update_id].material_id;
        if (material_id_of_update >= material_id) {
          material_id = material_id_of_update;
        }
      }

      MaterialUpdate update;
      update.material_id = material_id;

      memcpy(&update.material, object, sizeof(Material));

      __FAILURE_HANDLE(array_push(&scene->material_updates, &update));
    } break;
    case SCENE_ENTITY_INSTANCES: {
      // Get canonical new instance ID.
      uint32_t instances_count;
      __FAILURE_HANDLE(array_get_num_elements(scene->instances, &instances_count));

      uint32_t instance_id = instances_count;

      // If there are already instance updates queued, insert them at the end, otherwise, replace deleted instances.
      if (scene->flags & SCENE_DIRTY_FLAG_INSTANCES) {
        // There could be new instances queued in the instance_updates, we need to check for that.
        uint32_t instance_updates_count;
        __FAILURE_HANDLE(array_get_num_elements(scene->instance_updates, &instance_updates_count));

        for (uint32_t instance_update_id = 0; instance_update_id < instance_updates_count; instance_update_id++) {
          const uint32_t instance_id_of_update = scene->instance_updates[instance_update_id].instance_id;
          if (instance_id_of_update >= instance_id) {
            instance_id = instance_id_of_update;
          }
        }
      }
      else {
        // Search for the first deleted instance and replace it.
        for (uint32_t instance_list_id = 0; instance_list_id < instances_count; instance_list_id++) {
          if (scene->instances[instance_list_id].deleted) {
            instance_id = instance_list_id;
            break;
          }
        }
      }

      MeshInstanceUpdate update;
      update.instance_id = instance_id;

      memcpy(&update.instance, object, sizeof(MeshInstance));

      __FAILURE_HANDLE(array_push(&scene->instance_updates, &update));
    } break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Entity is not a list entity.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult scene_destroy(Scene** scene) {
  __CHECK_NULL_ARGUMENT(scene);

  __FAILURE_HANDLE(mutex_destroy(&(*scene)->mutex));

  __FAILURE_HANDLE(array_destroy(&(*scene)->materials));
  __FAILURE_HANDLE(array_destroy(&(*scene)->instances));

  __FAILURE_HANDLE(array_destroy(&(*scene)->material_updates));
  __FAILURE_HANDLE(array_destroy(&(*scene)->instance_updates));

  __FAILURE_HANDLE(host_free(&(*scene)->scratch_buffer));

  __FAILURE_HANDLE(host_free(scene));

  return LUMINARY_SUCCESS;
}
