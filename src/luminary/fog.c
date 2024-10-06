#include "fog.h"

#include "internal_error.h"

LuminaryResult fog_get_default(Fog* fog) {
  __CHECK_NULL_ARGUMENT(fog);

  fog->active           = false;
  fog->density          = 1.0f;
  fog->droplet_diameter = 10.0f;
  fog->height           = 500.0f;
  fog->dist             = 500.0f;

  return LUMINARY_SUCCESS;
}

#define __FOG_DIRTY(var)          \
  {                               \
    if (input->var != old->var) { \
      *dirty = true;              \
      return LUMINARY_SUCCESS;    \
    }                             \
  }

LuminaryResult fog_check_for_dirty(const Fog* input, const Fog* old, bool* dirty) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty);

  *dirty = false;

  __FOG_DIRTY(active);

  if (input->active) {
    __FOG_DIRTY(density);
    __FOG_DIRTY(droplet_diameter);
    __FOG_DIRTY(height);
    __FOG_DIRTY(dist);
  }

  return LUMINARY_SUCCESS;
}
