#ifndef LUMINARY_SCENE_H
#define LUMINARY_SCENE_H

#include "camera.h"
#include "mesh.h"
#include "mutex.h"
#include "sample_count.h"
#include "utils.h"

enum SceneEntity {
  SCENE_ENTITY_SETTINGS  = 0,
  SCENE_ENTITY_CAMERA    = 1,
  SCENE_ENTITY_OCEAN     = 2,
  SCENE_ENTITY_SKY       = 3,
  SCENE_ENTITY_CLOUD     = 4,
  SCENE_ENTITY_FOG       = 5,
  SCENE_ENTITY_PARTICLES = 6,
  SCENE_ENTITY_TOY       = 7,

  SCENE_ENTITY_MATERIALS = 8,
  SCENE_ENTITY_INSTANCES = 9,

  SCENE_ENTITY_COUNT,

  SCENE_ENTITY_GLOBAL_START = SCENE_ENTITY_SETTINGS,
  SCENE_ENTITY_GLOBAL_END   = SCENE_ENTITY_TOY,
  SCENE_ENTITY_GLOBAL_COUNT = SCENE_ENTITY_GLOBAL_END + 1,

  SCENE_ENTITY_LIST_START = SCENE_ENTITY_MATERIALS,
  SCENE_ENTITY_LIST_END   = SCENE_ENTITY_INSTANCES,
  SCENE_ENTITY_LIST_COUNT = SCENE_ENTITY_LIST_END + 1,
} typedef SceneEntity;

enum SceneEntityType {
  SCENE_ENTITY_TYPE_GLOBAL = 0,
  SCENE_ENTITY_TYPE_LIST   = 1,

  SCENE_ENTITY_TYPE_COUNT
} typedef SceneEntityType;

#define SCENE_ENTITY_TO_DIRTY(ENTITY) (1ull << ENTITY)

enum SceneDirtyFlag {
  // Generic
  SCENE_DIRTY_FLAG_OUTPUT      = 0x80000000ull,
  SCENE_DIRTY_FLAG_INTEGRATION = 0x40000000ull,
  SCENE_DIRTY_FLAG_BUFFERS     = 0x20000000ull,
  // Instances
  SCENE_DIRTY_FLAG_INSTANCES_LIGHT_LIST  = 0x01000000ull, /* Specifies that instances may have become lights or are no longer lights. */
  SCENE_DIRTY_FLAG_INSTANCES_LIGHT_DIRTY = 0x02000000ull, /* Specifies that instances that are lights, are dirty. */
  // Entities
  SCENE_DIRTY_FLAG_SETTINGS  = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_SETTINGS),
  SCENE_DIRTY_FLAG_CAMERA    = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_CAMERA),
  SCENE_DIRTY_FLAG_OCEAN     = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_OCEAN),
  SCENE_DIRTY_FLAG_SKY       = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_SKY),
  SCENE_DIRTY_FLAG_CLOUD     = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_CLOUD),
  SCENE_DIRTY_FLAG_FOG       = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_FOG),
  SCENE_DIRTY_FLAG_PARTICLES = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_PARTICLES),
  SCENE_DIRTY_FLAG_TOY       = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_TOY),
  SCENE_DIRTY_FLAG_MATERIALS = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_MATERIALS),
  SCENE_DIRTY_FLAG_INSTANCES = SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_INSTANCES)
} typedef SceneDirtyFlag;

typedef uint32_t SceneDirtyFlags;

extern size_t scene_entity_size[SCENE_ENTITY_GLOBAL_COUNT];

struct MaterialUpdate {
  Material material;
  uint32_t material_id;
} typedef MaterialUpdate;

struct MeshInstanceUpdate {
  MeshInstance instance;
  uint32_t instance_id;
} typedef MeshInstanceUpdate;

/*
 * The scene struct holds all the data that can be modified at runtime.
 * Scene data is subject to dirty checking.
 * Note that meshes and textures can only be added during runtime but
 * not be modified after the fact and are hence not present here.
 * Instances can be transformed and are hence present here.
 */
struct Scene {
  SceneDirtyFlags flags[SCENE_ENTITY_TYPE_COUNT];
  Mutex* mutex[SCENE_ENTITY_TYPE_COUNT];
  RendererSettings settings;
  Camera camera;
  Ocean ocean;
  Sky sky;
  Cloud cloud;
  Fog fog;
  Particles particles;
  Toy toy;
  ARRAY Material* materials;
  ARRAY MaterialUpdate* material_updates;
  ARRAY MeshInstance* instances;
  ARRAY MeshInstanceUpdate* instance_updates;
  void* scratch_buffer;
} typedef Scene;

LuminaryResult scene_create(Scene** scene);
LuminaryResult scene_lock(Scene* scene, SceneEntityType entity_mutex);
LuminaryResult scene_lock_all(Scene* scene);
LuminaryResult scene_get_dirty_flags(const Scene* scene, SceneDirtyFlags* flags);
LuminaryResult scene_get(Scene* scene, void* object, SceneEntity entity);
LuminaryResult scene_get_locking(Scene* scene, void* object, SceneEntity entity);
LuminaryResult scene_get_entry(Scene* scene, void* object, SceneEntity entity, uint32_t index);
LuminaryResult scene_get_entry_locking(Scene* scene, void* object, SceneEntity entity, uint32_t index);
LuminaryResult scene_unlock(Scene* scene, SceneEntityType entity_mutex);
LuminaryResult scene_unlock_all(Scene* scene);
LuminaryResult scene_update(Scene* scene, const void* object, SceneEntity entity);
LuminaryResult scene_update_force(Scene* scene, const void* object, SceneEntity entity);
LuminaryResult scene_update_entry(Scene* scene, const void* object, SceneEntity entity, uint32_t index);
LuminaryResult scene_get_entry_count(const Scene* scene, SceneEntity entity, uint32_t* count);
LuminaryResult scene_add_entry(Scene* scene, const void* object, SceneEntity entity);
LuminaryResult scene_apply_list_changes(Scene* scene, SceneEntity entity);
LuminaryResult scene_propagate_changes(Scene* scene, Scene* src);
LuminaryResult scene_destroy(Scene** scene);

#endif /* LUMINARY_SCENE_H */
