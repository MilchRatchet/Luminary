#ifndef LUMINARY_SCENE_H
#define LUMINARY_SCENE_H

#include "camera.h"
#include "instance.h"
#include "mutex.h"
#include "sample_count.h"
#include "utils.h"

enum SceneEntity { SCENE_ENTITY_SAMPLE_COUNT = 0, SCENE_ENTITY_CAMERA = 1, SCENE_ENTITY_INSTANCES = 2 } typedef SceneEntity;

#define SCENE_ENTITY_TO_DIRTY(ENTITY) (1u << ENTITY)

enum SceneDirtyFlag {
  SCENE_DIRTY_FLAG_SAMPLE_COUNT = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_SAMPLE_COUNT),
  SCENE_DIRTY_FLAG_CAMERA       = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_CAMERA),
  SCENE_DIRTY_FLAG_INSTANCES    = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_INSTANCES)
} typedef SceneDirtyFlag;

typedef uint64_t SceneDirtyFlags;

/*
 * The scene struct holds all the data that can be modified at runtime.
 * Scene data is subject to dirty checking.
 * Note that meshes and textures can only be added during runtime but
 * not be modified after the fact and are hence not present here.
 * Instances can be transformed and are hence present here.
 */
struct Scene {
  Mutex* mutex;
  SampleCountSlice sample_count;
  Camera camera;
  ARRAY Instance* instances;
  ARRAY InstanceUpdate* instance_updates;
} typedef Scene;

LuminaryResult scene_create(Scene** scene);
LuminaryResult scene_update(Scene* scene, const void* object, SceneEntity entity);
LuminaryResult scene_get(Scene* scene, void* object, SceneEntity entity);
LuminaryResult scene_destroy(Scene** scene);

#endif /* LUMINARY_SCENE_H */
