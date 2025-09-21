#include "fog.h"

#include "internal_error.h"
#include "scene.h"

LuminaryResult fog_get_default(Fog* fog) {
  __CHECK_NULL_ARGUMENT(fog);

  fog->active           = false;
  fog->density          = 1.0f;
  fog->droplet_diameter = 10.0f;
  fog->height           = 500.0f;
  fog->dist             = 500.0f;

  return LUMINARY_SUCCESS;
}

#define __FOG_CHECK_DIRTY(var)                                            \
  {                                                                       \
    if (input->var != old->var) {                                         \
      *dirty_flags = SCENE_DIRTY_FLAG_FOG | SCENE_DIRTY_FLAG_INTEGRATION; \
      return LUMINARY_SUCCESS;                                            \
    }                                                                     \
  }

LuminaryResult fog_check_for_dirty(const Fog* input, const Fog* old, uint32_t* dirty_flags) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty_flags);

  __FOG_CHECK_DIRTY(active);

  if (input->active) {
    __FOG_CHECK_DIRTY(density);
    __FOG_CHECK_DIRTY(droplet_diameter);
    __FOG_CHECK_DIRTY(height);
    __FOG_CHECK_DIRTY(dist);
  }

  return LUMINARY_SUCCESS;
}
